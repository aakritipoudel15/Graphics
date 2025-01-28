import glfw
import OpenGL.GL as gl
from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
import numpy as np
import csv
from models.logistic_regression import LogisticRegression

import os
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
filepath = os.path.join(base_dir, "../data/logistic_regression_data.csv")  # Move up one level



def normalize_data(X):
    """Normalize data to fit within the range [-1, 1]."""
    return 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1

def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def render_text(window, text, x, y):
    """Render text on the screen using GLFW."""
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glPushMatrix()
    gl.glLoadIdentity()
    width, height = glfw.get_window_size(window)
    gl.glOrtho(0, width, 0, height, -1, 1)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glPushMatrix()
    gl.glLoadIdentity()

    gl.glColor3f(0.0, 0.0, 0.0)  # Black color
    gl.glRasterPos2f(x, height - y)  # Flip y-coordinate for GLFW
    for character in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(character))

    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_MODELVIEW)

def load_data(filepath):
    """Load data from a CSV file."""
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)  # Skip header
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Labels
    return X, y

def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Logistic Regression Visualizer", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Set up orthographic projection
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-1, 1, -1, 1, -1, 1)  # Map coordinates to [-1, 1] range
    gl.glMatrixMode(gl.GL_MODELVIEW)

    # Load data
    X, y = load_data(filepath=filepath)

    # Normalize data
    X_normalized = normalize_data(X)
    y_normalized = y

    # Initialize model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    n_samples, n_features = X_normalized.shape
    model.weights = np.zeros(n_features)
    model.bias = 0

    # Training loop
    iteration = 0
    while not glfw.window_should_close(window):
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)  # Light theme background
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Draw grid
        gl.glColor3f(0.9, 0.9, 0.9)  # Light gray grid
        gl.glBegin(gl.GL_LINES)
        for i in np.arange(-1, 1.1, 0.2):
            gl.glVertex2f(i, -1)
            gl.glVertex2f(i, 1)
            gl.glVertex2f(-1, i)
            gl.glVertex2f(1, i)
        gl.glEnd()

        # Render data points
        gl.glPointSize(10.0)
        for i in range(len(X_normalized)):
            if y_normalized[i] == 0:
                gl.glColor3f(0.0, 0.0, 1.0)  # Blue for class 0
            else:
                gl.glColor3f(1.0, 0.0, 0.0)  # Red for class 1
            gl.glBegin(gl.GL_POINTS)
            gl.glVertex2f(X_normalized[i][0], X_normalized[i][1])
            gl.glEnd()

        # Perform one step of gradient descent
        if iteration < model.n_iterations:
            linear_model = np.dot(X_normalized, model.weights) + model.bias
            y_pred = sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X_normalized.T, (y_pred - y_normalized))
            db = (1 / n_samples) * np.sum(y_pred - y_normalized)
            model.weights -= model.learning_rate * dw
            model.bias -= model.learning_rate * db
            iteration += 1

        # Render decision boundary
        gl.glColor3f(0.0, 1.0, 0.0)  # Green decision boundary
        gl.glBegin(gl.GL_LINES)
        x_min, x_max = -1, 1
        y_min = -(model.weights[0] * x_min + model.bias) / model.weights[1]
        y_max = -(model.weights[0] * x_max + model.bias) / model.weights[1]
        gl.glVertex2f(x_min, y_min)
        gl.glVertex2f(x_max, y_max)
        gl.glEnd()

        # Display weights, bias, and iteration
        text = f"Iteration: {iteration}, Weights: {model.weights}, Bias: {model.bias:.4f}"
        render_text(window, text, 10, 20)  # Render text at (10, 20)

        # Render axis labels
        render_text(window, "Feature 1 (X-axis)", 350, 580)  # X-axis label
        render_text(window, "Feature 2 (Y-axis)", 10, 300)   # Y-axis label (rotated)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
