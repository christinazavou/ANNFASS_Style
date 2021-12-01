import tensorflow as tf
import os
from tensorboardX import SummaryWriter


# model_path = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/from_turing/trained_on_chairs/original/i16o128/batchdiv4_eval"
# batch_size = 4
# model_path = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/from_turing/trained_on_chairs/original/i16o128/batchdiv8_eval"
# batch_size = 8
# model_path = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/from_turing/trained_on_chairs/adain/i16o128/g32d32/b2_b32div_correctlog"
# batch_size = 32
model_path = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/from_turing/trained_on_chairs/adain/i16o128/g32d32/b_b32div_correctlog"
batch_size = 32


def change_summary(log="train_log"):
    writer = SummaryWriter(f"{model_path}/final_{log}")
    for tf_event_file in os.listdir(f"{model_path}/{log}"):
        for event in tf.train.summary_iterator(f"{model_path}/{log}/{tf_event_file}"):
            # print(event)
            # print(event.step)
            for v in event.summary.value:
                if 'grads' in v.tag:
                    continue
                if 'styles' in v.tag:
                    continue
                # print(v)
                writer.add_scalar(str(v.tag), float(v.simple_value), int(event.step) * batch_size)


if "train_log" in os.listdir(model_path):
    change_summary("train_log")
if "val_log" in os.listdir(model_path):
    change_summary("val_log")
