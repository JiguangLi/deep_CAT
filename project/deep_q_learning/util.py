class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5):
        """
        :param patience: how many consecutive epochs to wait before stopping when loss is
               not improving
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss < val_loss:
            self.best_loss = val_loss
            self.counter= 0
        elif self.best_loss >= val_loss:
            self.counter += 1
            if self.counter >= self.patience:
                print('INFO: Early stopping due to non-increasing average rewards')
                self.early_stop = True






