import PIL.Image
import google.generativeai as genai
import base64
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import pandas as pd
import numpy as np
import PyPDF2
from pdf2image import convert_from_path
import json
import csv
import uuid
from decimal import Decimal
from anthropic import Anthropic
from openai import OpenAI
import torch
from transformers import AutoModelForObjectDetection
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import os
import importlib


GOOGLE_API_KEY = genai.configure(api_key='YOUR_API_KEY')
CLAUDE_API_KEY = 'YOUR_API_KEY'
OPENAI_API_KEY = 'YOUR_API_KEY'

class table_extract():
    
    def __init__(self, model_name="microsoft/table-transformer-detection", max_size=800, threshold=0.5):
        self.model = AutoModelForObjectDetection.from_pretrained(model_name, revision="no_timm")
        self.model.config.id2label[len(self.model.config.id2label)] = "no object"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_size = max_size
        self.threshold = threshold
        self.project_id = "project_id"
        self.location = "location"
        
    

    class MaxResize(object):
        def __init__(self, max_size=800):
            self.max_size = max_size

        def __call__(self, image):
            width, height = image.size
            current_max_size = max(width, height)
            scale = self.max_size / current_max_size
            resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
            return resized_image
        
    
    #Postprocessing
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)


    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
    
    def outputs_to_objects(self, outputs, img_size, id2label):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})

        return objects
    

    def fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)
        return img


    def visualize_detected_tables(self, img, det_tables, out_path=None):
        plt.imshow(img, interpolation="lanczos")
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        ax = plt.gca()

        for det_table in det_tables:
            bbox = det_table['bbox']

            if det_table['label'] == 'table':
                facecolor = (1, 0, 0.45)
                edgecolor = (1, 0, 0.45)
                alpha = 0.3
                linewidth = 2
                hatch='//////'
            elif det_table['label'] == 'table rotated':
                facecolor = (0.95, 0.6, 0.1)
                edgecolor = (0.95, 0.6, 0.1)
                alpha = 0.3
                linewidth = 2
                hatch='//////'
            else:
                continue

            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                        edgecolor='none',facecolor=facecolor, alpha=0.1)
            ax.add_patch(rect)
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                        edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
            ax.add_patch(rect)
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                        edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
            ax.add_patch(rect)

        plt.xticks([], [])
        plt.yticks([], [])

        legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                                    label='Table', hatch='//////', alpha=0.3),
                            Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                                    label='Table (rotated)', hatch='//////', alpha=0.3)]
        plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
                        fontsize=10, ncol=2)
        plt.gcf().set_size_inches(10, 10)
        plt.axis('off')

        if out_path is not None:
          plt.savefig(out_path, bbox_inches='tight', dpi=150)

        return fig
    
    
    def objects_to_crops(self, img, tokens, objects, class_thresholds, padding=10):
        """
        Process the bounding boxes produced by the table detection model into
        cropped table images and cropped tokens.
        """

        table_crops = []
        for obj in objects:
            if obj['score'] < class_thresholds[obj['label']]:
                continue

            cropped_table = {}

            bbox = obj['bbox']
            bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

            cropped_img = img.crop(bbox)

            table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
            for token in table_tokens:
                token['bbox'] = [token['bbox'][0]-bbox[0],
                                token['bbox'][1]-bbox[1],
                                token['bbox'][2]-bbox[0],
                                token['bbox'][3]-bbox[1]]

            # If table is predicted to be rotated, rotate cropped image and tokens/words:
            if obj['label'] == 'table rotated':
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token['bbox']
                    bbox = [cropped_img.size[0]-bbox[3]-1,
                            bbox[0],
                            cropped_img.size[0]-bbox[1]-1,
                            bbox[2]]
                    token['bbox'] = bbox

            cropped_table['image'] = cropped_img
            cropped_table['tokens'] = table_tokens

            table_crops.append(cropped_table)

        return table_crops
    


    def main(self, image_path):
        image = PIL.Image.open(image_path).convert("RGB")
        
        detection_transform = transforms.Compose([
        self.MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pixel_values = detection_transform(image).unsqueeze(0)

        #Forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values)

        id2label = self.model.config.id2label
        id2label[len(self.model.config.id2label)] = "no object"

        #object detection
        objects = self.outputs_to_objects(outputs, image.size, id2label)

        # print(objects)

        tokens = []
        detection_class_thresholds = {
            "table": 0.5,
            "table rotated": 0.5,
            "no object": 10
        }
        crop_padding = 10
        
        tables_crops = self.objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=15)
        # print(len(objects))
        img_file_name = os.path.basename(image_path)
        cropped_table_paths = []

        for i in range(len(objects)):  
              
            cropped_table = tables_crops[i]['image'].convert("RGB")

            try:
                table_file_name = f"table_{i}_{img_file_name}"    
                cropped_table.save(f"tmp\\table_cropped_images\\{table_file_name}")
                print(f"Table {i} extracted from {img_file_name}")        #idhar file_name tha
                cropped_table_paths.append(f"tmp\\table_cropped_images\\{table_file_name}")
            except Exception as e:
                print(f"Error saving table {i}: {str(e)}")
                continue
            
        return cropped_table_paths
    






    def parse_pdf_and_convert_to_images(pdf_path, output_dir):
        # Open the PDF file
        with open(pdf_path, 'rb') as pdf_file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pages = len(pdf_reader.pages)
            # Loop through each page in the PDF
            for page_num in range(pages):
                # Extract the page
                page = pdf_reader.getPage(page_num)
                
                # Convert the page to an image
                images = convert_from_path(pdf_path, single_file=True, first_page=page_num + 1, last_page=page_num + 1)
                
                # Save the image
                image_path = f"{output_dir}/page_{page_num + 1}.png"
                images[0].save(image_path, 'PNG')
                
                print(f"Page {page_num + 1} converted and saved as {image_path}")


    

    def verifier(self, file_path):
        client = Anthropic(api_key=CLAUDE_API_KEY)
        response = client.messages.create(
          model="claude-3-sonnet-20240229",
          max_tokens=1024,
          temperature=0.0,
          messages=[
              {
                "role": "user",
                "content": [
                    {
                      "type": "image",
                      "source": {
                          "type": "base64",
                          "data": Path(__file__).parent.joinpath(file_path),
                          "media_type": "image/png"
                      }
                    },
                    {
                      "type": "text",
                      "text": """
                            As a PROFESSIONAL TABLE DETECTION TOOL, your task is to accurately detect whether the provided image contains any table in it. Any data that is in table format is accecpted by the tool.
                            
                            Chain of thought process that you should follow:
                            **STEP1:** Scan the provided image and check if it contains any data that is in tabular format.
                            **STEP2:** If image contains a table or anything that seems to be in a table format return 'TABLE_DETECTED'.
                            **STEP3:** Else return 'NO_TABLE_DETECTED'
                            

                            THREE GOLDEN RULES: 
                              1. You are restricted to only return one of two words i.e. 'TABLE_DETECTED' or 'NO_TABLE_DETECTED' and nothing else.
                              2. Follow the Chain of thought process.
                              3. Do not forget the above 2 rules.
                              
                            NOTE: Skip the preamble; go striaght into your one word response.
                            Response:
                            """
                    }
                ]
                
              }
          ]
        )

        return response
    

    def openai_image_summary(self, file_path):
        client = OpenAI(api_key=OPENAI_API_KEY)
        with open(file_path, "rb") as image_file:
            base64_img = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = """
        Summarize the following image.
        Analyze the image and provide a concise description of its content. Identify the main objects, scenes, or activities depicted. 

        NOTE: Set the preamble as 'IMAGE SUMMARY:'; then go straight into response. RETURN IMAGE SUMMARY AT ALL COSTS.
        """
        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                },
                },
            ],
            }
        ],
        temperature=0.0,
        max_tokens=1024,
        )

        return response.choices[0].message.content
    


    

    def openai_summarizer(self, file_path):
        client = OpenAI(api_key=OPENAI_API_KEY)
        with open(file_path, "rb") as image_file:
            base64_img = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = """
        As a PROFESSIONAL IMAGE OCR SCANNER for table extraction, your task is to accurately detect the boundaries in the provided table image, extract the text, and structure it into a precise summary representing the table data.
        Your task is to take the unstructured text provided and convert it into a well-organized summary that is understandable. Identify the main entities, attributes, or categories mentioned in the text and use them in the summary. 

        TWO GOLDEN RULES:
        1. You should always return the structured table data in the form of a precise summary and NOTHING ELSE.
        2. Do not forget the above rule 1.
        
        NOTE: Set the preamble as 'SUMMARY:'; then go straight into response. RETURN SUMMARY AT ALL COSTS. 

        <example> “SUMMARY: Two sets of experiments were conducted with standard gas volume concentrations of 100 ppm, 500 ppm, and 1000 ppm. In the first set, the experimental results were 102.9 ppm, 508.3 ppm, and 986.5 ppm respectively, with relative errors of 2.9%, 1.66%, and -1.35%, and absolute errors of 2.9 ppm, 8.3 ppm, and 13.5 ppm. In the second set, the experimental results were 104.9 ppm, 521.3 ppm, and 971.5 ppm respectively, with relative errors of 4.9%, 4.26%, and -2.85%, and absolute errors of 4.9 ppm, 21.3 ppm, and 28.5 ppm.” </example>
        """
        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                },
                },
            ],
            }
        ],
        temperature=0.0,
        max_tokens=1024,
        )

        return response.choices[0].message.content
    

    def encode_image_to_base64(self, file_path):
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_image}"
        

    def openai_multiple_imgs(self, img_path_list):
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = """
        Describe the image provided.
        """
            
        image_urls = [self.encode_image_to_base64(file_path) for file_path in img_path_list]
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt    #change here ,
                        },
                        *[
                            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
                        ],
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=1300,
        )

        return response.choices[0].message.content
    

    def openai_test(self, file_path):
        client = OpenAI(api_key=OPENAI_API_KEY)
        with open(file_path, "rb") as image_file:
            base64_img = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt ="""
                As a PROFESSIONAL IMAGE OCR SCANNER for table extraction, your task is to accurately detect the boundaries in the provided table image, extract the text, and structure it into a json format representing the table data.
                Your task is to take the unstructured text provided and convert it into a well-organized table format using JSON. Identify the main entities, attributes, or categories mentioned in the text and use them as keys in the JSON object. Then, extract the relevant information from the text and populate the corresponding values in the JSON object. Ensure that the data is accurately represented and properly formatted within the JSON structure. The resulting JSON table should provide a clear, structured overview of the information presented in the original text.

                TWO GOLDEN RULES: 
                1. You should always return the structured table data in the form of a JSON format and NOTHING ELSE. 
                2. Do not forget the above 2 rules.
                              
                NOTE: Skip the preamble; go straight into response and you must generate text without adding newline character in the output, markdown output is prohibited. RETURN JSON OBJECT AT ALL COSTS.
                *Data Structure:* The structured table data should maintain the original table structure with rows and columns appropriately represented for easy manipulation and further processing, such as dumping it to json file.

                <example>
                [
                    {
                        "Standard gas volume concentration (ppm)": [100, 500, 1000],
                        "Experimental result (ppm)": [102.9, 508.3, 986.5],
                        "The relative error (%)": [2.9, 1.66, -1.35],
                        "The absolute error (ppm)": [2.9, 8.3, 13.5]
                    },
                    {
                        "Standard gas volume concentration (ppm)": [100.0, 500.0, 1000.0],
                        "Experimental result (ppm)": [104.9, 521.3, 971.5],
                        "The relative error (%)": [4.9, 4.26, -2.85],
                        "The absolute error (ppm)": [4.9, 21.3, 28.5]
                    }
                ]
                </example>
                """
        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                },
                },
            ],
            }
        ],
        temperature=0.0,
        max_tokens=1024,
        )

        return response.choices[0].message.content
    


    def img_to_dict_openai(self, file_path):
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt ="""
                As a PROFESSIONAL IMAGE OCR SCANNER for table extraction, your task is to accurately detect the boundaries in the provided table image, extract the text, and structure it into a json format representing the table data.
                Your task is to take the unstructured text provided and convert it into a well-organized table format using JSON. Identify the main entities, attributes, or categories mentioned in the text and use them as keys in the JSON object. Then, extract the relevant information from the text and populate the corresponding values in the JSON object. Ensure that the data is accurately represented and properly formatted within the JSON structure. The resulting JSON table should provide a clear, structured overview of the information presented in the original text.

                TWO GOLDEN RULES: 
                1. You should always return the structured table data in the form of a JSON format and NOTHING ELSE. 
                2. Do not forget the above 2 rules.
                              
                NOTE: Skip the preamble; go straight into response and you must generate text without adding newline character in the output. RETURN JSON OBJECT AT ALL COSTS.
                *Data Structure:* The structured table data should maintain the original table structure with rows and columns appropriately represented for easy manipulation and further processing, such as dumping it to json file.
                """
        response = client.chat.completions.create(
          model="gpt-4-1106-vision-preview",
          max_tokens=1024,
          temperature=0.0,
          messages=[
                        {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": Path(__file__).parent.joinpath(file_path),
                            },
                            },
                        ],
                        }
                    ]
        )

        return response
    


    




    def img_to_dict_claude(self, file_path):
        client = Anthropic(api_key=CLAUDE_API_KEY)
        response = client.messages.create(
          model="claude-3-opus-20240229",
          max_tokens=1024,
          temperature=0.0,
          messages=[
              {
                "role": "user",
                "content": [
                    {
                      "type": "image",
                      "source": {
                          "type": "base64",
                          "data": Path(__file__).parent.joinpath(file_path),
                          "media_type": "image/png"
                      }
                    },
                    {
                      "type": "text",
                      "text": """
                            As a PROFESSIONAL IMAGE OCR SCANNER for table extraction, your task is to accurately detect the boundaries in the provided table image, extract the text, and structure it into a json format representing the table data.

                            **TASK1:** Extract all textual information from the provided table image.
                            **TASK2:** Convert the extracted text into a Python dictionary or a dictionary of lists containing the structured table data.

                            THREE GOLDEN RULES: 
                              1. You should always return the structured table data in the form of a JSON format (without ) and NOTHING ELSE.
                              2. If you sense/detect that the provided image is not a table then return 'NONE' and nothing else. 
                              3. Do not forget the above 2 rules.
                              
                            NOTE: Skip the preamble; go straight into response and you must generate text without adding newline character in the output. Markdown output is prohibited. RETURN JSON OBJECT AT ALL COSTS.
                            **Data Structure:** The structured table data should maintain the original table structure with rows and columns appropriately represented for easy manipulation and further processing, such as dumping it to json file.

                            <example>
                            [
                                {
                                    "Standard gas volume concentration (ppm)": [100, 500, 1000],
                                    "Experimental result (ppm)": [102.9, 508.3, 986.5],
                                    "The relative error (%)": [2.9, 1.66, -1.35],
                                    "The absolute error (ppm)": [2.9, 8.3, 13.5]
                                },
                                {
                                    "Standard gas volume concentration (ppm)": [100.0, 500.0, 1000.0],
                                    "Experimental result (ppm)": [104.9, 521.3, 971.5],
                                    "The relative error (%)": [4.9, 4.26, -2.85],
                                    "The absolute error (ppm)": [4.9, 21.3, 28.5]
                                }
                            ]
                            </example>
                            """
                    }
                ]
                
              }
          ]
        )

        return response
    



    def main_main(self, pdf_path):

    
        output_dir = "tmp\pdf_img_table"
        self.parse_pdf_and_convert_to_images(pdf_path, output_dir)

        # Get the list of files in the directory
        image_files = os.listdir(output_dir)
        table_data = []
        false_table_count = 0

        print(image_files)
        # Iterate through each image file and process it with table_obj.main
        for file_name in image_files:
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(output_dir, file_name)
                    cropped_img_paths = self.main(image_path)
                    
                    if cropped_img_paths:
                        for cropped_img_path in cropped_img_paths:
                            response_verified = self.verifier(file_path=cropped_img_path)
                            for block_verified in response_verified.content:
                                if block_verified.type == 'text':
                                    response_verified_text = block_verified.text
                            try:
                                if response_verified_text == 'TABLE_DETECTED':
                    ## Uncomment 1 or 2 or 3 accordingly


                                ## 1) Use for OpenAI inference
                                    response = self.openai_test(file_path=cropped_img_path)
                                    table_data.append(response)


                                ## 2) Use for Claude inference
                                    # response = self.img_to_dict_claude(file_path=cropped_img_path)
                                    # print(response)
                                    # for block in response.content:
                                    #     if block.type == "text":
                                    #         table_data.append(block.text)


                                ## 3) Use for table summary OpenAI inference
                                    # response = self.openai_summarizer(file_path=cropped_img_path)
                                    # table_data.append(response)



                                elif response_verified_text == 'NO_TABLE_DETECTED':
                                    false_table_count += 1
                                print(f"Number of false tables detected: {false_table_count}")
                                
                            except Exception as e:
                                print(f"Error occured while detecting the tables, error: {e}")
                    else:
                        print(f"No tables extracted from {file_name}")

        ## putting to json response to json file
        try:
            table_data_json = []
            # Loop through each JSON string, load it, and append to the combined JSON data list
            for json_str in table_data:
                json_obj = json.loads(json_str)
                table_data_json.append(json_obj)
            with open('output.json', 'w') as f:
                json.dump(table_data_json, f, indent=4)
            print(f"Table data saved to json file: output.json")
        except Exception as e:
            print(f"Error occurred while saving data to JSON file: {e}")
        print(table_data)
        print(table_data_json)


        ## putting summary json file
        try:
            #Loop through each summary string, load it, and append to the table_data_summary list
            print(table_data)
        except Exception as e:
            print(f"Table suummary iteration failed")
        
        
        return table_data
    
    


 
    

    




if __name__ == "__main__":
   table_obj = table_extract()
  # TESTING: image to dict using claude
  #  response = table_extract.img_to_dict_claude("test23.png")
  #  for block in response.content:
  #     if block.type == "text":
  #        print(block.text)
   

  # TESTING: table extraction
  #  table_obj.main("else11_page-0003.jpg")
  #  table_obj.main("else11_page-0008.jpg")
  #  table_obj.main("j.ijleo.2016.01.051 (1)_page-0003.jpg")
   






   
#    pdf_path = "tmp\j.ijleo.2016.01.051 (1).pdf"
   pdf_path = "j.chemphys.2016.12.005.pdf"
#    pdf_path = "malki.pdf"
#    pdf_path = "Paper 5 0973005220964688 paper URban haat e copy[1].pdf"
#    pdf_path = "tmp\Vitamin D effects on cardiovascular diseases.pdf"
#    pdf_path = "tmp\j.phpro.2014.09.002.pdf"
   output_dir = "tmp\pdf_img_table"
#    parse_pdf_and_convert_to_images(pdf_path, output_dir)

   # Get the list of files in the directory
   image_files = os.listdir(output_dir)
   table_data = []
   false_table_count = 0

   
   # Iterate through each image file and process it with table_obj.main
   for file_name in image_files:
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(output_dir, file_name)
            cropped_img_paths = table_obj.main(image_path)
            
            if cropped_img_paths:
                for cropped_img_path in cropped_img_paths:
                    response_verified = table_obj.verifier(file_path=cropped_img_path)
                    for block_verified in response_verified.content:
                        if block_verified.type == 'text':
                            response_verified_text = block_verified.text
                    try:
                        if response_verified_text == 'TABLE_DETECTED':
            ## Uncomment 1 or 2 or 3 accordingly


                        ## 1) Use for OpenAI inference
                            # response = table_obj.openai_test(file_path=cropped_img_path)
                            # table_data.append(response)


                        ## 2) Use for Claude inference
                            # response = table_obj.img_to_dict_claude(file_path=cropped_img_path)
                            # print(response)
                            # for block in response.content:
                            #     if block.type == "text":
                            #         table_data.append(block.text)


                        ## 3) Use for table summary OpenAI inference
                            response = table_obj.openai_summarizer(file_path=cropped_img_path)
                            print(response)
                            table_data.append(response)


                        elif response_verified_text == 'NO_TABLE_DETECTED':
                            false_table_count += 1
                        print(f"Number of false tables detected: {false_table_count}")
                        
                    except Exception as e:
                        print(f"Error occured while detecting the tables, error: {e}")
            else:
                print(f"No tables extracted from {file_name}")

## putting to json response to json file
   try:
      table_data_json = []
    # Loop through each JSON string, load it, and append to the combined JSON data list
      for json_str in table_data:
          json_obj = json.loads(json_str)
          table_data_json.append(json_obj)
      with open('output.json', 'w') as f:
          json.dump(table_data_json, f, indent=4)
      print(f"Table data saved to json file: output.json")
   except Exception as e:
      print(f"Error occurred while saving data to JSON file: {e}")
   print(table_data)
   print(table_data_json)


## putting summary json file
   try:
    #Loop through each summary string, load it, and append to the table_data_summary list
       for summary_str in table_data:
           print(f"Talbe summary: {summary_str}")
   except Exception as e:
       print(f"Table suummary iteration failed")
       


