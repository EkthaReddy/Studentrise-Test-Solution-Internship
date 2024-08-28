# Studentrise-Test-Solution-Internship

## Handwritten Digit Recognizer
To develop a handwritten digit recognizer using the MNIST dataset, you can follow these steps:

1. **Set Up the Environment:**
   Ensure you have the necessary libraries installed. You will need TensorFlow/Keras for the neural network, and Tkinter or another GUI library for the interface.
   ```bash
   pip install tensorflow numpy matplotlib
   ```

2. **Load and Preprocess the MNIST Dataset:**
   Load the dataset, normalize the pixel values, and split the data into training and testing sets.
   ```python
   import tensorflow as tf
   from tensorflow.keras.datasets import mnist

   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0

   x_train = x_train.reshape(-1, 28, 28, 1)
   x_test = x_test.reshape(-1, 28, 28, 1)
   ```

3. **Build the CNN Model:**
   Define a convolutional neural network using Keras.
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

   model = Sequential([
       Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
       MaxPooling2D(pool_size=(2, 2)),
       Conv2D(64, kernel_size=(3, 3), activation='relu'),
       MaxPooling2D(pool_size=(2, 2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(10, activation='softmax')
   ])

   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   ```

4. **Train the Model:**
   Train the model with the training data.
   ```python
   model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
   ```

5. **Evaluate the Model:**
   Evaluate the model's performance on the test data.
   ```python
   test_loss, test_acc = model.evaluate(x_test, y_test)
   print(f'Test accuracy: {test_acc}')
   ```

6. **Create the GUI:**
   Use Tkinter to create a simple GUI where users can draw digits.
   ```python
   import tkinter as tk
   from PIL import Image, ImageDraw, ImageOps
   import numpy as np

   class DigitRecognizerApp(tk.Tk):
       def __init__(self, model):
           super().__init__()
           self.title("Handwritten Digit Recognizer")
           self.canvas = tk.Canvas(self, width=200, height=200, bg='white')
           self.canvas.pack()
           self.button_predict = tk.Button(self, text="Predict", command=self.predict_digit)
           self.button_predict.pack()
           self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
           self.button_clear.pack()
           self.model = model
           self.image = Image.new('L', (200, 200), 255)
           self.draw = ImageDraw.Draw(self.image)
           self.canvas.bind('<B1-Motion>', self.paint)

       def paint(self, event):
           x1, y1 = (event.x - 5), (event.y - 5)
           x2, y2 = (event.x + 5), (event.y + 5)
           self.canvas.create_oval(x1, y1, x2, y2, fill='black')
           self.draw.ellipse([x1, y1, x2, y2], fill='black')

       def predict_digit(self):
           image = self.image.resize((28, 28)).convert('L')
           image = ImageOps.invert(image)
           image = np.array(image) / 255.0
           image = image.reshape(1, 28, 28, 1)
           prediction = self.model.predict(image)
           digit = np.argmax(prediction)
           tk.messagebox.showinfo("Prediction", f"The digit is: {digit}")

       def clear_canvas(self):
           self.canvas.delete("all")
           self.draw.rectangle([0, 0, 200, 200], fill='white')

   if __name__ == "__main__":
       app = DigitRecognizerApp(model)
       app.mainloop()
   
