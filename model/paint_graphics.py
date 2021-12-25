import matplotlib.pyplot as plt


class Paint:

    def __init__(self, h):
        self.h = h
        self.paint_mae()
        self.paint_loss()
    
    def paint_mae(self):
        plt.plot(self.h.history['mae'])
        plt.plot(self.h.history['val_mae'])
        plt.title('Precision del modelo')
        plt.ylabel('Precision')
        plt.xlabel('epocas')
        plt.legend(['Entrenamiento', 'test'], loc='upper left')
        plt.savefig("precision_modelo.jpg")
        plt.close()


    def paint_loss(self):
        plt.plot(self.h.history['loss'])
        plt.plot(self.h.history['val_loss'])
        plt.title('Perdidas del modelo')
        plt.ylabel('perdidas')
        plt.xlabel('epocas')
        plt.legend(['Entrenamiento', 'test'], loc='upper left')
        plt.savefig("perdidas_modelo.jpg")
        plt.close()

