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

from PIL import Image, ImageOps

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    
class BingImageGrabber:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                #"num_of_images": ("INT", {"default": 1,}),
                "search_term": ("STRING", {"default": "dog"}),
                "cache_search": ("BOOLEAN", {"default": True},),
                "cache_images": ("BOOLEAN", {"default": True},),
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
        
    def test(self, search_term, cache_search, cache_images):

        iterations = 1

        bing_txt = search_term.lower().strip()
        
        print(bing_txt)
        
        bing_txt.translate(dict.fromkeys(map(ord, u"/\&:+:%")))
        
        texthash = str(hash(bing_txt))

        urlopenheader={ 'User-Agent' : 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}
        foundimage=None
        
        if not os.path.exists("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"):
            os.makedirs("./ComfyUI/custom_nodes/ConCarneNode/searchcache/")
        
        if (os.path.isfile("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+texthash+".searchcache")) and cache_search:
            print ("ok")
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
                    print("loop finished " + str(current))
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
        
        if not os.path.exists("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+bing_txt) and cache_images:
            os.makedirs("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+bing_txt)          
                
        for tries in range(10):                           
        
            url = totallinks[random.randint(0, len(totallinks) - 1)]
            
            filename = posixpath.basename(url).split('?')[0] #Strip GET parameters from filename
            name, ext = os.path.splitext(filename)
            name = name[:36].strip()
            name = name + str(hash(name))[0:4]
            filename = (name + ext).strip()
            
            if os.path.isfile("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+bing_txt+"/"+filename) and cache_images:
                foundimage = Image.open("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+bing_txt+"/"+filename)
                print ('Loading image from cache')
                break
            
            print ("downloading " + url)
            
            try:
                request=urllib.request.Request(url,None,urlopenheader)
                image_data=urllib.request.urlopen(request).read()
            except:
                continue
                        
            if not imghdr.what(None, image_data):
                print('Invalid image, not loading')
                continue
            
            foundimage = Image.open( io.BytesIO(image_data))
            
            if cache_images:
                imagefile=open(os.path.join("./ComfyUI/custom_nodes/ConCarneNode/searchcache/"+bing_txt+"/", filename),'wb')
                imagefile.write(image_data)
                imagefile.close()
            
            break          
           

        image = foundimage
        
        #image = foundimage.resize((512,512))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = pil2tensor(image).unsqueeze(0)
        
        print (image.size())
        
        return (image)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "BingImageGrabber": BingImageGrabber
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "BingImageGrabber": "Bing Image Grabber"
}
