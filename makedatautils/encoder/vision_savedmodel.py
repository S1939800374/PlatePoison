#将.pb文件load到session中，导出到.pbtxt可视化
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
 
model = 'makedatautils/encoder/imagenet'
export_dir = 'makedatautils/encoder/txt'
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.saved_model.load(sess,[tf.compat.v1.saved_model.tag_constants.SERVING],model)
 
    builder = tf.compat.v1.saved_model.Builder(export_dir)
    builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING])
    builder.save(as_text=True)