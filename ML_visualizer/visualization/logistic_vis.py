import glfw
import OpenGL.GL as gl
from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
import numpy as np
import csv
from models.logistic_regression import LogisticRegression
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(base_dir, "../data/logistic_regression_data.csv")

def normalize_data(X):
    return 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def render_text(window, text, x, y, size=18):
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glPushMatrix()
    gl.glLoadIdentity()
    width, height = glfw.get_window_size(window)
    gl.glOrtho(0, width, 0, height, -1, 1)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glPushMatrix()
    gl.glLoadIdentity()

    gl.glColor3f(1.0, 1.0, 1.0)  # White text
    gl.glRasterPos2f(x, height - y)
    for character in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(character))

    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_MODELVIEW)

def draw_axes(min_val, max_val):
    """Draw coordinate axes with arrows."""
    gl.glColor3f(1.0, 1.0, 1.0)  # White axes
    
    # X axis
    gl.glBegin(gl.GL_LINES)
    gl.glVertex2f(min_val, 0)
    gl.glVertex2f(max_val, 0)
    # X axis arrow
    gl.glVertex2f(max_val, 0)
    gl.glVertex2f(max_val - 0.1, 0.05)
    gl.glVertex2f(max_val, 0)
    gl.glVertex2f(max_val - 0.1, -0.05)
    gl.glEnd()
    
    # Y axis
    gl.glBegin(gl.GL_LINES)
    gl.glVertex2f(0, min_val)
    gl.glVertex2f(0, max_val)
    # Y axis arrow
    gl.glVertex2f(0, max_val)
    gl.glVertex2f(0.05, max_val - 0.1)
    gl.glVertex2f(0, max_val)
    gl.glVertex2f(-0.05, max_val - 0.1)
    gl.glEnd()

def draw_grid(min_val, max_val, step):
    gl.glColor3f(0.2, 0.2, 0.2)  # Dark gray grid
    gl.glBegin(gl.GL_LINES)
    for i in np.arange(min_val, max_val + step, step):
        if abs(i) > 1e-10:  # Skip the axis lines
            gl.glVertex2f(i, min_val)
            gl.glVertex2f(i, max_val)
            gl.glVertex2f(min_val, i)
            gl.glVertex2f(max_val, i)
    gl.glEnd()

def draw_sigmoid(weights, bias, x_range):
    gl.glColor3f(1.0, 0.5, 0.0)  # Orange curve
    gl.glBegin(gl.GL_LINE_STRIP)
    for x in np.linspace(x_range[0], x_range[1], 200):
        z = weights[0] * x + bias
        y = sigmoid(z)
        gl.glVertex2f(x, y)
    gl.glEnd()

def main():
    if not glfw.init():
        return

    # Create window with a 2:1 aspect ratio
    window = glfw.create_window(1400, 700, "Logistic Regression Visualizer", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Load and prepare data
    X, y = np.loadtxt(filepath, delimiter=',', skiprows=1)[:, :-1], np.loadtxt(filepath, delimiter=',', skiprows=1)[:, -1]
    X_normalized = normalize_data(X)
    
    # Initialize model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    n_samples, n_features = X_normalized.shape
    model.weights = np.zeros(n_features)
    model.bias = 0

    iteration = 0
    while not glfw.window_should_close(window):
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # Dark background
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        width, height = glfw.get_window_size(window)
        
        # Left viewport - Decision Boundary
        gl.glViewport(0, 0, width , height*2)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-1.2, 1.2, -1.2, 1.2, -1, 1)  # Slightly larger range for labels
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
        draw_grid(-1, 1, 0.2)
        draw_axes(-1, 1)
        
        # Draw data points
        gl.glPointSize(8.0)
        for i in range(len(X_normalized)):
            if y[i] == 0:
                gl.glColor3f(0.3, 0.7, 1.0)  # Light blue for class 0
            else:
                gl.glColor3f(1.0, 0.3, 0.3)  # Light red for class 1
            gl.glBegin(gl.GL_POINTS)
            gl.glVertex2f(X_normalized[i][0], X_normalized[i][1])
            gl.glEnd()

        # Draw decision boundary
        if not np.all(model.weights == 0):
            gl.glColor3f(0.0, 1.0, 0.5)  # Green decision boundary
            gl.glBegin(gl.GL_LINES)
            x_min, x_max = -1.2, 1.2
            if model.weights[1] != 0:
                y_min = -(model.weights[0] * x_min + model.bias) / model.weights[1]
                y_max = -(model.weights[0] * x_max + model.bias) / model.weights[1]
                gl.glVertex2f(x_min, y_min)
                gl.glVertex2f(x_max, y_max)
            gl.glEnd()

        # Right viewport - Sigmoid Function
        gl.glViewport(width , 0, width , height*2)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-6, 6, -0.2, 1.2, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
        draw_grid(-6, 6, 1.0)
        draw_axes(-6, 6)
        
        # Draw sigmoid function
        draw_sigmoid(model.weights, model.bias, (-6, 6))
        
        # Draw current decision point
        gl.glPointSize(10.0)
        gl.glColor3f(1.0, 1.0, 0.0)  # Yellow point
        gl.glBegin(gl.GL_POINTS)
        decision_point = -model.bias / model.weights[0] if not np.all(model.weights == 0) else 0
        gl.glVertex2f(decision_point, sigmoid(0))
        gl.glEnd()

        # Perform gradient descent
        if iteration < model.n_iterations:
            linear_model = np.dot(X_normalized, model.weights) + model.bias
            y_pred = sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X_normalized.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            model.weights -= model.learning_rate * dw
            model.bias -= model.learning_rate * db
            iteration += 1

        # Render text information
        # Left viewport text
        render_text(window, "Decision Boundary View", width // 4 - 80, 30)
        render_text(window, "Feature X₁", width // 4 - 40, height - 20)
        render_text(window, "Feature X₂", 20, height // 2)
        
        # Right viewport text
        render_text(window, "Sigmoid Function View", 3 * width // 4 - 80, 30)
        render_text(window, "z = w₁x₁ + w₂x₂ + b", 3 * width // 4 - 80, 60)
        render_text(window, "z", 3 * width // 4 - 40, height - 20)
        render_text(window, "σ(z)", width // 2 + 20, height // 2)

        # Model parameters
        render_text(window, f"Iteration: {iteration}", 10, 90)
        render_text(window, f"Weights: [{model.weights[0]:.2f}, {model.weights[1]:.2f}]", 10, 120)
        render_text(window, f"Bias: {model.bias:.2f}", 10, 150)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()


# import glfw
# import OpenGL.GL as gl
# from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
# import numpy as np
# import csv
# from models.logistic_regression import LogisticRegression
# import os

# base_dir = os.path.dirname(os.path.abspath(__file__))
# filepath = os.path.join(base_dir, "../data/logistic_regression_data.csv")

# class InteractiveLogisticRegression:
#     def __init__(self, window_width=1400, window_height=700):
#         self.window_width = window_width
#         self.window_height = window_height
#         self.X = np.empty((0, 2))
#         self.y = np.empty(0)
#         self.current_class = 0  # Toggle between 0 and 1
#         self.model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
#         self.model.weights = np.zeros(2)
#         self.model.bias = 0
#         self.iteration = 0
#         self.training = False
#         self.window = None
#         self.last_click_time = 0
#         self.click_cooldown = 0.1  # Seconds between allowed clicks

#     def normalize_data(self, X):
#         if len(X) == 0:
#             return X
#         return 2 * (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-10) - 1

#     def screen_to_world_coordinates(self, x, y):
#         """Convert screen coordinates to world coordinates."""
#         viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
#         if x < viewport[2] // 2:  # Left viewport
#             wx = (x / (viewport[2] // 2) * 2.4) - 1.2
#             wy = ((viewport[3] - y) / viewport[3] * 2.4) - 1.2
#             return wx, wy
#         return None, None  # Click was in right viewport

#     def mouse_button_callback(self, window, button, action, mods):
#         if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
#             current_time = glfw.get_time()
#             if current_time - self.last_click_time < self.click_cooldown:
#                 return
#             self.last_click_time = current_time
            
#             x, y = glfw.get_cursor_pos(window)
#             wx, wy = self.screen_to_world_coordinates(x, y)
            
#             if wx is not None:  # Click was in left viewport
#                 new_point = np.array([[wx, wy]])
#                 self.X = np.vstack([self.X, new_point])
#                 self.y = np.append(self.y, self.current_class)
#                 self.training = True  # Start training when new point is added
#                 self.iteration = 0  # Reset training
#                 self.model.weights = np.zeros(2)  # Reset weights
#                 self.model.bias = 0  # Reset bias
        
#         elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
#             self.current_class = 1 - self.current_class  # Toggle class

#     def key_callback(self, window, key, scancode, action, mods):
#         if key == glfw.KEY_SPACE and action == glfw.PRESS:
#             self.training = not self.training  # Toggle training
#         elif key == glfw.KEY_C and action == glfw.PRESS:
#             self.X = np.empty((0, 2))  # Clear all points
#             self.y = np.empty(0)
#             self.model.weights = np.zeros(2)
#             self.model.bias = 0
#             self.iteration = 0
#             self.training = False

#     def render_text(self, text, x, y):
#         gl.glMatrixMode(gl.GL_PROJECTION)
#         gl.glPushMatrix()
#         gl.glLoadIdentity()
#         gl.glOrtho(0, self.window_width, 0, self.window_height, -1, 1)
#         gl.glMatrixMode(gl.GL_MODELVIEW)
#         gl.glPushMatrix()
#         gl.glLoadIdentity()

#         gl.glColor3f(1.0, 1.0, 1.0)
#         gl.glRasterPos2f(x, self.window_height - y)
#         for character in text:
#             glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(character))

#         gl.glPopMatrix()
#         gl.glMatrixMode(gl.GL_PROJECTION)
#         gl.glPopMatrix()
#         gl.glMatrixMode(gl.GL_MODELVIEW)

#     def draw_axes(self, min_val, max_val):
#         gl.glColor3f(1.0, 1.0, 1.0)
#         gl.glBegin(gl.GL_LINES)
#         gl.glVertex2f(min_val, 0)
#         gl.glVertex2f(max_val, 0)
#         gl.glVertex2f(0, min_val)
#         gl.glVertex2f(0, max_val)
#         gl.glEnd()

#     def draw_grid(self, min_val, max_val, step):
#         gl.glColor3f(0.2, 0.2, 0.2)
#         gl.glBegin(gl.GL_LINES)
#         for i in np.arange(min_val, max_val + step, step):
#             if abs(i) > 1e-10:
#                 gl.glVertex2f(i, min_val)
#                 gl.glVertex2f(i, max_val)
#                 gl.glVertex2f(min_val, i)
#                 gl.glVertex2f(max_val, i)
#         gl.glEnd()

#     def draw_sigmoid(self, x_range):
#         gl.glColor3f(1.0, 0.5, 0.0)
#         gl.glBegin(gl.GL_LINE_STRIP)
#         for x in np.linspace(x_range[0], x_range[1], 200):
#             z = self.model.weights[0] * x + self.model.bias
#             y = 1 / (1 + np.exp(-z))
#             gl.glVertex2f(x, y)
#         gl.glEnd()

#     def run(self):
#         if not glfw.init():
#             return

#         self.window = glfw.create_window(self.window_width, self.window_height, 
#                                        "Interactive Logistic Regression", None, None)
#         if not self.window:
#             glfw.terminate()
#             return

#         glfw.make_context_current(self.window)
#         glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
#         glfw.set_key_callback(self.window, self.key_callback)

#         while not glfw.window_should_close(self.window):
#             gl.glClearColor(0.0, 0.0, 0.0, 1.0)
#             gl.glClear(gl.GL_COLOR_BUFFER_BIT)

#             # Left viewport - Decision Boundary
#             gl.glViewport(0, 0, self.window_width , self.window_height )
#             gl.glMatrixMode(gl.GL_PROJECTION)
#             gl.glLoadIdentity()
#             gl.glOrtho(-1.2, 1.2, -1.2, 1.2, -1, 1)
#             gl.glMatrixMode(gl.GL_MODELVIEW)

#             self.draw_grid(-1, 1, 0.2)
#             self.draw_axes(-1, 1)

#             # Draw data points
#             if len(self.X) > 0:
#                 gl.glPointSize(8.0)
#                 for i in range(len(self.X)):
#                     if self.y[i] == 0:
#                         gl.glColor3f(0.3, 0.7, 1.0)
#                     else:
#                         gl.glColor3f(1.0, 0.3, 0.3)
#                     gl.glBegin(gl.GL_POINTS)
#                     gl.glVertex2f(self.X[i][0], self.X[i][1])
#                     gl.glEnd()

#             # Draw decision boundary
#             if not np.all(self.model.weights == 0) and len(self.X) > 0:
#                 gl.glColor3f(0.0, 1.0, 0.5)
#                 gl.glBegin(gl.GL_LINES)
#                 x_min, x_max = -1.2, 1.2
#                 if self.model.weights[1] != 0:
#                     y_min = -(self.model.weights[0] * x_min + self.model.bias) / self.model.weights[1]
#                     y_max = -(self.model.weights[0] * x_max + self.model.bias) / self.model.weights[1]
#                     gl.glVertex2f(x_min, y_min)
#                     gl.glVertex2f(x_max, y_max)
#                 gl.glEnd()

#             # Right viewport - Sigmoid Function
#             gl.glViewport(self.window_width , 0, self.window_width , self.window_height)
#             gl.glMatrixMode(gl.GL_PROJECTION)
#             gl.glLoadIdentity()
#             gl.glOrtho(-6, 6, -0.2, 1.2, -1, 1)
#             gl.glMatrixMode(gl.GL_MODELVIEW)

#             self.draw_grid(-6, 6, 1.0)
#             self.draw_axes(-6, 6)
#             self.draw_sigmoid((-6, 6))

#             # Training step
#             if self.training and len(self.X) > 0 and self.iteration < self.model.n_iterations:
#                 X_norm = self.normalize_data(self.X)
#                 linear_model = np.dot(X_norm, self.model.weights) + self.model.bias
#                 y_pred = 1 / (1 + np.exp(-linear_model))
#                 dw = (1 / len(self.X)) * np.dot(X_norm.T, (y_pred - self.y))
#                 db = (1 / len(self.X)) * np.sum(y_pred - self.y)
#                 self.model.weights -= self.model.learning_rate * dw
#                 self.model.bias -= self.model.learning_rate * db
#                 self.iteration += 1

#             # Render text information
#             self.render_text(f"Current Class: {self.current_class} ({'Blue' if self.current_class == 0 else 'Red'})", 10, 30)
#             self.render_text("Right-click to toggle class", 10, 60)
#             self.render_text("Space to pause/resume training", 10, 90)
#             self.render_text("'C' to clear all points", 10, 120)
#             self.render_text(f"Training: {'Active' if self.training else 'Paused'}", 10, 150)
#             self.render_text(f"Iteration: {self.iteration}", 10, 180)
#             self.render_text(f"Weights: [{self.model.weights[0]:.2f}, {self.model.weights[1]:.2f}]", 10, 210)
#             self.render_text(f"Bias: {self.model.bias:.2f}", 10, 240)

#             # View labels
#             self.render_text("Decision Boundary View", self.window_width // 4 - 80, 30)
#             self.render_text("Feature X₁", self.window_width // 4 - 40, self.window_height - 20)
#             self.render_text("Feature X₂", 20, self.window_height // 2)
#             self.render_text("Sigmoid Function View", 3 * self.window_width // 4 - 80, 30)
#             self.render_text("z", 3 * self.window_width // 4 - 40, self.window_height - 20)
#             self.render_text("σ(z)", self.window_width // 2 + 20, self.window_height // 2)

#             glfw.swap_buffers(self.window)
#             glfw.poll_events()

#         glfw.terminate()


# def main():
#     glutInit()  # Ensure GLUT is initialized before usage
#     visualizer = InteractiveLogisticRegression()
#     visualizer.run()

# if __name__ == "__main__":
#     main()