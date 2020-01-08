import tensorflow as tf
assert tf.__version__[0] == '1'


def export_tf1(session, in_tnsr_fullname, out_tnsr_fullname, export_dir='./export'):
    assert isinstance(in_tnsr_fullname, str)
    assert isinstance(out_tnsr_fullname, str)

    in_tnsr_name = in_tnsr_fullname.split(':')[0]
    out_tnsr_name = out_tnsr_fullname.split(':')[0]

    output_graph_def = tf.graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), [out_tnsr_name])
    tf.train.write_graph(output_graph_def, './tmp_write_graph', name='tmp_write_graph', as_text=False)

    tf.reset_default_graph()
    graph_def = tf.GraphDef()
    with tf.gfile.GFile('./tmp_write_graph/tmp_write_graph', "rb") as f:
        graph_def.ParseFromString(f.read())
    [out] = tf.import_graph_def(graph_def, name="", return_elements=[out_tnsr_fullname])
    g = out.graph

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.Session(graph=g) as sess:
        input_signatures, output_signatures = {in_tnsr_name: g.get_tensor_by_name(in_tnsr_fullname)}, {
            out_tnsr_name: g.get_tensor_by_name(out_tnsr_fullname)}

        signature = tf.saved_model.signature_def_utils.predict_signature_def(input_signatures, output_signatures)

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
            clear_devices=True
        )

    builder.save()
