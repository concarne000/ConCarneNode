import copy
import math
import os
import random
import sys
import traceback
import shlex
import os, urllib.request, re, threading, posixpath, urllib.parse, argparse, socket, time, hashlib, pickle, signal, imghdr, io
import numpy as np
import torch
import hashlib

from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoTokenizer

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    
def center_crop(img, new_width=None, new_height=None):        

    widthcenter = img.width / 2
    heightcenter = img.height / 2

    center_cropped_img = img.crop((widthcenter - (new_width/2), heightcenter - (new_height/2), widthcenter + (new_width/2), heightcenter + (new_height/2)))

    return center_cropped_img    
    
class BingImageGrabber:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "search_term": ("STRING", {"default": "dog"}),
                "num_of_images": ("INT", {"default": 1}),
                "cache_search": ("BOOLEAN", {"default": True},),
                "cache_images": ("BOOLEAN", {"default": True},),
                "use_number_of_links": ("INT", {"default": -1}),
             },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "ConCarne"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
        
    def test(self, search_term, num_of_images, cache_search, cache_images, use_number_of_links):

        iterations = 1

        bing_txt = search_term.lower().strip()
                
        bing_txt.translate(dict.fromkeys(map(ord, u"/\&:+:%")))
        
        texthash = str(hashlib.md5(bing_txt.encode()).hexdigest())

        urlopenheader={ 'User-Agent' : 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}
        foundimage=None
        
        if not os.path.exists("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"):
            os.makedirs("./ComfyUI/custom_nodes/ConCarneNode/searchcache/")
        
        if (os.path.isfile("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+texthash+".searchcache")) and cache_search:
            #print ("Loading links from cache")
            text_file = open("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+texthash+".searchcache", "r")
            totallinks = text_file.readlines()
            text_file.close()
            
        else:
            current = 0
            last = ''
            totallinks = []
            
            done = False
            
            while done == False:
                time.sleep(0.5)
                request_url='https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(bing_txt) + '&first=' + str(current) + '&count=35&adlt=off'
                request=urllib.request.Request(request_url,None,headers=urlopenheader)
                response=urllib.request.urlopen(request)
                html = response.read().decode('utf8')
                links = re.findall('murl&quot;:&quot;(.*?)&quot;',html)
                
                if links[-1] == last or current > 20:
                    done = True
                    #print("loop finished " + str(current))
                    break

                current += 1
                last = links[-1]
                totallinks.extend(links)
            
            if cache_search:
            
                text_file = open("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+texthash+".searchcache", "w")
                
                for l in totallinks:
                    text_file.write(l + "\n")

                text_file.close()          

        images = []
        all_prompts = []
        infotexts = []
        
        foundimage = None
        
        if not os.path.exists("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+texthash) and cache_images:
            os.makedirs("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+texthash)          
        
        if (use_number_of_links > 0):
            totallinks = totallinks[0:use_number_of_links]
        
        random.shuffle(totallinks)
        
        for index in range(len(totallinks)):
            
            if (len(images) >= num_of_images):
                break
            
            for tries in range(10):                           
            
                url = totallinks[index]
                
                filename = posixpath.basename(url).split('?')[0] #Strip GET parameters from filename
                name, ext = os.path.splitext(filename)
                name = name[:36].strip()
                name = name + str(hash(name))[0:4]
                filename = (name + ext).strip()
                
                filenamehash = hashlib.md5( filename.encode() ).hexdigest()
                
                if os.path.isfile("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+texthash+"/"+filenamehash+".jpg") and cache_images:
                    foundimage = Image.open("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+texthash+"/"+filenamehash+".jpg")
                    #print ('Loading image from cache')
                    
                else:
                
                    #print ("downloading " + url)
                    
                    try:
                        request=urllib.request.Request(url,None,urlopenheader)
                        image_data=urllib.request.urlopen(request, timeout=3).read()
                    except:
                        continue
                                
                    if not imghdr.what(None, image_data):
                        print('Invalid image, not loading')
                        continue
                    
                    foundimage = Image.open( io.BytesIO(image_data))
                    
                    if cache_images:
                        #imagefile=open(os.path.join("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+texthash+"/", filenamehash+".jpg"),'wb')
                        #imagefile.write(image_data)
                        #imagefile.close()
                        foundimage = foundimage.convert("RGB")
                        foundimage.save(os.path.join("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+texthash+"/", filenamehash+".jpg"))

                image = foundimage
                
                minsize = min(foundimage.width, foundimage.height)
                
                if (num_of_images > 1):
                
                    foundimage = center_crop(foundimage, minsize, minsize)
                
                    image = foundimage.resize((1024,1024))
                    
                else:
                
                    image = foundimage
                    
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")
                image = pil2tensor(image).unsqueeze(0)
                
                images.append(image)
                
                print("Image number " + str(len(images)) + " downloaded")
                
                break          
 
        
        
        #print (image.size())
        
        return ( torch.cat(images, dim=0), )

class Zephyr:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A list of interesting subjects for photographs are as follows:"}),
             },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "test"

    CATEGORY = "ConCarne"

    #@classmethod
    #def IS_CHANGED(cls, **kwargs):
    #    return float("NaN")
    
    def test(self, prompt):

        tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')
        model = AutoModelForCausalLM.from_pretrained(
            'stabilityai/stablelm-zephyr-3b',
            trust_remote_code=True,
            device_map="auto"
        )

        prompt = [{'role': 'user', 'content': prompt}]
        inputs = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        tokens = model.generate(
            inputs.to(model.device),
            max_new_tokens=512,
            temperature=0.8,
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )

        text = tokenizer.decode(tokens[0], skip_special_tokens=False)   
        text = text.split("<|assistant|>")[-1]
        text = text.replace("<|endoftext|>","")
        
        compiledtext = ""
        
        for line in text.splitlines():
            if len(line) > 3:
                compiledtext = compiledtext + line + "\n"

        return (compiledtext,)

class Hermes:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A list of interesting subjects for photographs are as follows:"}),
             },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "test"

    CATEGORY = "ConCarne"

    #@classmethod
    #def IS_CHANGED(cls, **kwargs):
    #    return float("NaN")
    
    def test(self, prompt):

        tokenizer = AutoTokenizer.from_pretrained('TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ')
        model = AutoModelForCausalLM.from_pretrained(
            'TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ',
            device_map="auto",
            trust_remote_code=False,
            revision="main")

        prompt = prompt
        prompt_template=f'''<|im_start|>system
            {system_message}<|im_end|>
            <|im_start|>user
            {prompt}<|im_end|>
            <|im_start|>assistant
            '''
            
        input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)

        return (tokenizer.decode(output[0]),)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "BingImageGrabber": BingImageGrabber,
    "Zephyr": Zephyr,
    "Hermes": Hermes
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "BingImageGrabber": "Bing Image Grabber",
    "Zephyr": "Zephyr Chat",
    "Hermes": "Hermes Chat"
}
