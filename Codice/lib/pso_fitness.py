from sklearn.metrics import accuracy_score
import numpy as np

class obj_fun:
    def __init__(self, scores_val_mods, y_true, mode = "opt"):
        self.scores_path = scores_val_mods[0]
        self.scores_rad = scores_val_mods[1]
        self.scores_sem = scores_val_mods[2]

        self.y_true = y_true

        self.mode = mode
    
    def solve(self, weights):
        if self.mode == "opt":
            x = weights[0]
            y = weights[1]
            z = weights[2]
            
            scores = self.scores_path*x + self.scores_rad*y + self.scores_sem*z
            y_pred = np.argmax(scores, axis=0)
            acc = accuracy_score(self.y_true, y_pred)
            err = 1-acc
            return err
        elif self.mode == "pred":
            x = weights[0]
            y = weights[1]
            z = weights[2]
            
            scores = self.scores_path*x + self.scores_rad*y + self.scores_sem*z
            y_pred = np.argmax(scores, axis=0)
            acc = accuracy_score(self.y_true, y_pred)
            err = 1-acc
            return y_pred