class ASREvaluationObserver:
    def __init__(self, save_path):
        self.poison_num = self.poison_num_success = 0
        self.save_path = save_path
 
    @property
    def asr(self):
        if self.poison_num == 0:
            return 0
        
        return self.poison_num_success / self.poison_num
 
    def __call__(self, pred, batch, additional):
        poison_gt = batch["labels"]
        poison_mask = poison_gt != batch["not_poisoned_labels"]
        self.poison_num_success += (poison_mask.logical_and(pred == poison_gt)).sum().item()
        self.poison_num += poison_mask.sum().item()
 
    def __enter__(self):
        self.poison_num = self.poison_num_success = 0
        return self
 
    def __exit__(self, type, value, traceback):
        print(f'ASR: {(self.asr * 100):.2f}%')
        with open(self.save_path, 'a') as f:
            f.write(f'ASR: {(self.asr * 100):.2f}%\n')