import os
import json
import numpy as np
import mxnet as mx
from mxnet import gluon 

def model_fn(model_dir):
    
    #load pretrained model
    model = gluon.SymbolBlock.imports(
        '%s/model/model-symbol.json' % model_dir,
        ['data'],
        '%s/model/model-0000.params' % model_dir,
    ) 
    return model
    
def transform_fn(model, data, content_type, output_content_type):
    """
    Transform incoming requests.
    """
    #decode json string into numpy array
    data = json.loads(data)
    
    #convert to MXNet NDArray
    image = mx.nd.array(data)
    
    #check if GPUs area available
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    
    #load image on the right context
    image = image.as_in_context(ctx)
    
    #inference
    class_IDs, scores, bounding_boxes = model(image)
    
    #convert results to a list
    result = [class_IDs.asnumpy().tolist(), scores.asnumpy().tolist(), bounding_boxes.asnumpy().tolist()]
    
    #decode result as json string
    response_body = json.dumps(result)
    
    return response_body, output_content_type
