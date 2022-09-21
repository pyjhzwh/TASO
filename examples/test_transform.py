# import torch

# dummy_input = torch.randn(10, 3, 224, 224, device="cuda")

# class TransformLayout(torch.nn.Module):
#     def forward(self, x):
#         y = (x + x).to(memory_format=torch.channels_last)
#         return y

# input_names = [ "input_1" ]
# output_names = [ "output1" ]
# model = TransformLayout().cuda()
# torch.onnx.export(model, dummy_input, "transformLayout.onnx", verbose=True, input_names=input_names, output_names=output_names)


import taso as ts

graph = ts.new_graph()
input = graph.new_input(dims=(1,256,28,28))
t1 = graph.transform(input=input, src_layout="NCHW", dst_layout="NCHW")
new_graph = ts.optimize(graph, alpha=1.0, budget=1000)
