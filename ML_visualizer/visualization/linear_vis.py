# import glfw
# import OpenGL.GL as gl
# from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
# import numpy as np
# from models.linear_regression import LinearRegression

# def normalize_data(X, y):
#     """Normalize data to fit within the range [-1, 1]."""
#     X_normalized = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1
#     y_normalized = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1
#     return X_normalized, y_normalized

# def render_text(window, text, x, y):
#     """Render text on the screen using GLFW."""
#     gl.glMatrixMode(gl.GL_PROJECTION)
#     gl.glPushMatrix()
#     gl.glLoadIdentity()
#     width, height = glfw.get_window_size(window)
#     gl.glOrtho(0, width, 0, height, -1, 1)
#     gl.glMatrixMode(gl.GL_MODELVIEW)
#     gl.glPushMatrix()
#     gl.glLoadIdentity()

#     gl.glColor3f(1.0, 1.0, 1.0)  # White color
#     gl.glRasterPos2f(x, height - y)  # Flip y-coordinate for GLFW
#     for character in text:
#         glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(character))

#     gl.glPopMatrix()
#     gl.glMatrixMode(gl.GL_PROJECTION)
#     gl.glPopMatrix()
#     gl.glMatrixMode(gl.GL_MODELVIEW)

# def main():
#     if not glfw.init():
#         return

#     window = glfw.create_window(800, 600, "Linear Regression Visualizer", None, None)
#     if not window:
#         glfw.terminate()
#         return

#     glfw.make_context_current(window)

#     # Set up orthographic projection
#     gl.glMatrixMode(gl.GL_PROJECTION)
#     gl.glLoadIdentity()
#     gl.glOrtho(-1, 1, -1, 1, -1, 1)  # Map coordinates to [-1, 1] range
#     gl.glMatrixMode(gl.GL_MODELVIEW)

#     # Example data
#     X = np.array([[1], [2], [3], [4], [5]])
#     y = np.array([1, 3, 2, 3, 5])

#     # Normalize data
#     X_normalized, y_normalized = normalize_data(X, y)

#     # Initialize model
#     model = LinearRegression(learning_rate=0.01, n_iterations=1000)
#     n_samples, n_features = X_normalized.shape
#     model.weights = np.zeros(n_features)
#     model.bias = 0

#     # Training loop
#     iteration = 0
#     while not glfw.window_should_close(window):
#         gl.glClear(gl.GL_COLOR_BUFFER_BIT)

#         # Render data points
#         gl.glColor3f(1.0, 1.0, 1.0)  # White color
#         gl.glPointSize(5.0)  # Set point size
#         gl.glBegin(gl.GL_POINTS)
#         for i in range(len(X_normalized)):
#             gl.glVertex2f(X_normalized[i][0], y_normalized[i])
#         gl.glEnd()

#         # Perform one step of gradient descent
#         if iteration < model.n_iterations:
#             y_pred = np.dot(X_normalized, model.weights) + model.bias
#             dw = (1 / n_samples) * np.dot(X_normalized.T, (y_pred - y_normalized))
#             db = (1 / n_samples) * np.sum(y_pred - y_normalized)
#             model.weights -= model.learning_rate * dw
#             model.bias -= model.learning_rate * db
#             iteration += 1

#         # Render regression line
#         gl.glColor3f(0.0, 1.0, 0.0)  # Green color
#         gl.glBegin(gl.GL_LINES)
#         x_min, x_max = X_normalized.min(), X_normalized.max()
#         y_min, y_max = model.predict(np.array([[x_min]]))[0], model.predict(np.array([[x_max]]))[0]
#         gl.glVertex2f(x_min, y_min)
#         gl.glVertex2f(x_max, y_max)
#         gl.glEnd()

#         # Display weights, bias, and iteration
#         text = f"Iteration: {iteration}, Weights: {model.weights[0]:.4f}, Bias: {model.bias:.4f}"
#         render_text(window, text, 10, 20)  # Render text at (10, 20)

#         glfw.swap_buffers(window)
#         glfw.poll_events()

#     glfw.terminate()

# if __name__ == "__main__":
#     main()

import glfw
import OpenGL.GL as gl
from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
import numpy as np
from models.linear_regression import LinearRegression

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

    gl.glColor3f(1.0, 1.0, 1.0)  # White color
    gl.glRasterPos2f(x, height - y)  # Flip y-coordinate for GLFW
    for character in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(character))

    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_MODELVIEW)

def screen_to_gl_coordinates(window, x, y):
    """Convert screen coordinates to OpenGL coordinates."""
    width, height = glfw.get_window_size(window)
    gl_x = (2.0 * x / width) - 1.0
    gl_y = 1.0 - (2.0 * y / height)
    return gl_x, gl_y

def find_nearest_point(points, x, y, threshold=0.1):
    """Find the nearest point within threshold distance."""
    if not points:
        return None
    
    distances = [(i, np.sqrt((p[0] - x)**2 + (p[1] - y)**2)) 
                for i, p in enumerate(points)]
    nearest_idx, distance = min(distances, key=lambda x: x[1])
    
    return nearest_idx if distance < threshold else None

def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Linear Regression Visualizer", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glutInit()

    # Set up orthographic projection
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-1, 1, -1, 1, -1, 1)  # Map coordinates to [-1, 1] range
    gl.glMatrixMode(gl.GL_MODELVIEW)

    # Initialize model and variables
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    points = []
    iteration = 0
    dragging = False
    selected_point = None
    training_active = False

    def mouse_button_callback(window, button, action, mods):
        nonlocal points, iteration, dragging, selected_point, model
        
        if button == glfw.MOUSE_BUTTON_LEFT:
            x, y = glfw.get_cursor_pos(window)
            gl_x, gl_y = screen_to_gl_coordinates(window, x, y)
            
            if action == glfw.PRESS:
                selected_point = find_nearest_point(points, gl_x, gl_y)
                
                if selected_point is None:
                    points.append([gl_x, gl_y])
                    # Reset model when adding new points
                    if len(points) >= 2:
                        X = np.array([[p[0]] for p in points])
                        y = np.array([p[1] for p in points])
                        model.weights = np.zeros(1)
                        model.bias = 0
                        iteration = 0
                else:
                    dragging = True
            
            elif action == glfw.RELEASE:
                dragging = False
                selected_point = None

    def cursor_position_callback(window, x, y):
        nonlocal points, selected_point, dragging, iteration, model
        if dragging and selected_point is not None:
            gl_x, gl_y = screen_to_gl_coordinates(window, x, y)
            points[selected_point] = [gl_x, gl_y]
            if len(points) >= 2:
                # Reset model when moving points
                model.weights = np.zeros(1)
                model.bias = 0
                iteration = 0

    def key_callback(window, key, scancode, action, mods):
        nonlocal training_active, points, iteration, model
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            training_active = not training_active
        elif key == glfw.KEY_C and action == glfw.PRESS:
            points.clear()
            model.weights = np.zeros(1)
            model.bias = 0
            iteration = 0
            training_active = False

    # Set callbacks
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_key_callback(window, key_callback)

    # Main loop
    while not glfw.window_should_close(window):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Only proceed with training if we have at least 2 points
        if len(points) >= 2:
            # Convert points to numpy arrays
            X = np.array([[p[0]] for p in points])
            y = np.array([p[1] for p in points])
            
            # Normalize data
            X_normalized, y_normalized = normalize_data(X, y)
            n_samples = len(points)

            # Perform one step of gradient descent if training is active
            if training_active and iteration < model.n_iterations:
                y_pred = np.dot(X_normalized, model.weights) + model.bias
                dw = (1 / n_samples) * np.dot(X_normalized.T, (y_pred - y_normalized))
                db = (1 / n_samples) * np.sum(y_pred - y_normalized)
                model.weights -= model.learning_rate * dw
                model.bias -= model.learning_rate * db
                iteration += 1

            # Render regression line
            gl.glColor3f(0.0, 1.0, 0.0)  # Green color
            gl.glBegin(gl.GL_LINES)
            x_min, x_max = X_normalized.min(), X_normalized.max()
            y_min = model.predict(np.array([[x_min]]))[0]
            y_max = model.predict(np.array([[x_max]]))[0]
            gl.glVertex2f(x_min, y_min)
            gl.glVertex2f(x_max, y_max)
            gl.glEnd()

        # Render points
        gl.glColor3f(1.0, 1.0, 1.0)  # White color
        gl.glPointSize(5.0)  # Set point size
        gl.glBegin(gl.GL_POINTS)
        for point in points:
            gl.glVertex2f(point[0], point[1])
        gl.glEnd()

        # Display information
        if len(points) >= 2:
            status = f"Iteration: {iteration}, "
            status += f"Weights: {model.weights[0]:.4f}, "
            status += f"Bias: {model.bias:.4f}, "
            status += f"Training: {'Active' if training_active else 'Paused'}"
            render_text(window, status, 10, 20)

        # Display instructions
        instructions = [
            "Click to add points",
            "Drag points to move them",
            "Space to start/pause training",
            "C to clear all points"
        ]
        for i, text in enumerate(instructions):
            render_text(window, text, 10, 50 + i * 20)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()