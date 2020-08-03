# convert .h5 to .pb
import tensorflow as tf

#def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#    from tensorflow.python.framework.graph_util import convert_variables_to_constants
#    graph = session.graph
#    with graph.as_default():
#        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#        output_names = output_names or []
#        output_names += [v.op.name for v in tf.global_variables()]
#        input_graph_def = graph.as_graph_def()
#        if clear_devices:
#            for node in input_graph_def.node:
#                node.device = ""
#        frozen_graph = convert_variables_to_constants(session, input_graph_def,
#                                                      output_names, freeze_var_names)
#        return frozen_graph

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        print('output_names:',output_names)
        #freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        #print('freeze_var_names:',freeze_var_names)
        #output_names = output_names or []
        #output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        #frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names)
        return frozen_graph

'''
sess = K.get_session()
frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])

from tensorflow.python.framework import graph_io

output_path='.'
pb_model_name='5_trained_model.pb'
graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)
'''
