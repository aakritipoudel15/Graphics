import glfw
import OpenGL.GL as gl
from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
import numpy as np
from models.linear_regression import LinearRegression
import os

def normalize_data(X, y):
    """Normalize data to fit within the range [-1, 1]."""
    X_normalized = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1
    y_normalized = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1
    return X_normalized, y_normalized

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

    gl.glColor3f(0.0, 0.0, 0.0)  # Black color for text
    gl.glRasterPos2f(x, height - y)  # Flip y-coordinate for GLFW
    for character in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(character))

    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_MODELVIEW)

def draw_grid():
    """Draw grid and axes."""
    gl.glColor3f(0.8, 0.8, 0.8)  # Light gray for grid
    gl.glBegin(gl.GL_LINES)
    for i in np.arange(-1, 1.1, 0.2):
        gl.glVertex2f(i, -1)
        gl.glVertex2f(i, 1)
        gl.glVertex2f(-1, i)
        gl.glVertex2f(1, i)
    gl.glEnd()

    # Draw X and Y axes
    gl.glColor3f(0.0, 0.0, 0.0)  # Black color
    gl.glBegin(gl.GL_LINES)
    gl.glVertex2f(-1, 0)
    gl.glVertex2f(1, 0)
    gl.glVertex2f(0, -1)
    gl.glVertex2f(0, 1)
    gl.glEnd()

def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Linear Regression Visualizer", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Set up orthographic projection
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-1, 1, -1, 1, -1, 1)  # Map coordinates to [-1, 1] range
    gl.glMatrixMode(gl.GL_MODELVIEW)

    # Example data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 3, 2, 3, 5])

    # Normalize data
    X_normalized, y_normalized = normalize_data(X, y)

    # Initialize model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    n_samples, n_features = X_normalized.shape
    model.weights = np.zeros(n_features)
    model.bias = 0

    # Training loop
    iteration = 0
    while not glfw.window_should_close(window):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)  # Set light background

        # Draw grid and axes
        draw_grid()

        # Render data points
        gl.glColor3f(0.0, 0.0, 1.0)  # Blue color for points
        gl.glPointSize(5.0)  # Set point size
        gl.glBegin(gl.GL_POINTS)
        for i in range(len(X_normalized)):
            gl.glVertex2f(X_normalized[i][0], y_normalized[i])
        gl.glEnd()

        # Perform one step of gradient descent
        if iteration < model.n_iterations:
            y_pred = np.dot(X_normalized, model.weights) + model.bias
            dw = (1 / n_samples) * np.dot(X_normalized.T, (y_pred - y_normalized))
            db = (1 / n_samples) * np.sum(y_pred - y_normalized)
            model.weights -= model.learning_rate * dw
            model.bias -= model.learning_rate * db
            iteration += 1

        # Render regression line
        gl.glColor3f(1.0, 0.0, 0.0)  # Red color for line
        gl.glBegin(gl.GL_LINES)
        x_min, x_max = X_normalized.min(), X_normalized.max()
        y_min, y_max = model.predict(np.array([[x_min]]))[0], model.predict(np.array([[x_max]]))[0]
        gl.glVertex2f(x_min, y_min)
        gl.glVertex2f(x_max, y_max)
        gl.glEnd()

        # Display weights, bias, and iteration
        text = f"Iteration: {iteration}, Weights: {model.weights[0]:.4f}, Bias: {model.bias:.4f}"
        render_text(window, text, 10, 20)  # Render text at (10, 20)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
