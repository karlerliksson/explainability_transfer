# T5 training utility functions

def get_best_checkpoint(metric_scores, eval_task, eval_metric, reverse=True):
    metric_scores = filter(lambda s: s["eval_task"] == eval_task, metric_scores)
    sorted_metric_scores = sorted(metric_scores,
                                  key=lambda s: s[eval_metric],
                                  reverse=reverse)
    return sorted_metric_scores[0]["step"]

def cleanup_checkpoints(metric_scores, task, metrics, checkpoints_dir):
    """
    Removes all checkpoints in checkpoints_dir that are not optimal w.r.t any of the metrics
    """
    best_checkpoints = set([get_best_checkpoint(metric_scores, task, metric) for metric in metrics])
    best_checkpoint_names = ["model-{}.checkpoint".format(ckpt) for ckpt in best_checkpoints]
    checkpoints_files = [f for f in checkpoints_dir.iterdir() if f.is_file() and f.suffix == ".checkpoint"]

    try:
        for f in checkpoints_files:
            if f.name not in best_checkpoint_names:
                f.unlink()
        return True

    except IOError:
        return False
