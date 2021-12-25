import csv
import numpy as np

class Save:

    def __init__(self, test, predictions):
        y_true = test
        # y_true = np.zeros((np.size(predictions)))
        y_pred = np.array([np.argmax(x) for x in predictions])
		
		
        with open('evaluation_rn50.csv', 'w') as f1:
            writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )

            for i in range(0, len(y_true)):
                writer.writerow([y_pred[i], y_true[i]])
                writer.writerow([predictions[i], y_true[i]])