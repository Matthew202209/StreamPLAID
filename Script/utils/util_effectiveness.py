import os

def save_effectiveness_metrics(file, task, result_list):
    # write csv header once
    if not os.path.isfile(file):
        with open(file, "w") as metric_file:
            metric_file.write("task,RR,nDCG,Success\n")
    with open(file, "a") as metric_file:
        metric_file.write(str(task) + "," + str(result_list['RR']) + "," + str(result_list['nDCG']) + "," + str(result_list['Success']) + "\n")

def save_effectiveness_metrics_update(file, streaming_step, task, result_list):
    # write csv header once
    if not os.path.isfile(file):
        with open(file, "w") as metric_file:
            metric_file.write("Task, StreamingStep, RR, nDCG, Success\n")
    with open(file, "a") as metric_file:
        metric_file.write(
             str(task) + "," +str(streaming_step) + "," + str(result_list['RR']) + "," + str(result_list['nDCG']) + "," + str(result_list['Success']) + "\n")