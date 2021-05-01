import sys
import mxnet
from mxnet import gluon
from json import load as json_load 
from json import dumps as json_dumps

sym_json = json_load(open('mx-mod-symbol.json'))
sym_json_string = json_dumps(sym_json)
model = gluon.nn.SymbolBlock(
    outputs=mxnet.sym.load_json(sym_json_string),
    inputs=mxnet.sym.var('data'))
model.load_parameters('mx-mod-0000.params', allow_missing=True)
model.initialize()

def mxnet_predict(x, model=model):
    return model(mxnet.nd.array([x]))[0].asscalar()

def handler(event, context):
    output = mxnet_predict(float(event["param"]))
    return f"{output}"
    
if __name__ == "__main__":
    output = handler({"param": 42}, {})
    print(output)