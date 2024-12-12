from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from glfw.GLFW import *
import sys


class LineDrawing:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.frequencies = []
        self.algorithm = 'dda'

    def plot_pixel(self, x, y, color,size=5.0):
        """Plot a pixel at the given coordinates (screen space)."""
        glColor3f(*color)  # Set color
        glPointSize(size)  # Set point size
        glBegin(GL_POINTS)
        glVertex2f(x, y)
        glEnd()

    def draw_line(self, x1, y1, x2, y2, color):
        """Draw a line between two points with the given color."""
        glColor3f(*color)
        glLineWidth(2.0)  # Set line width
        glBegin(GL_LINES)
        glVertex2f(x1, y1)
        glVertex2f(x2, y2)
        glEnd()

    def dda_line(self, x1, y1, x2, y2, color):
        """Digital Differential Analyzer (DDA) Line Algorithm."""
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        x_inc = dx / steps
        y_inc = dy / steps
        x, y = x1, y1
        for _ in range(int(steps) + 1):
            self.plot_pixel(x, y, color)
            x += x_inc
            y += y_inc

    def bresenham_line(self, x1, y1, x2, y2, color):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = int(x1), int(y1)
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != int(x2):
                self.plot_pixel(int(x), int(y), color)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != int(y2):
                self.plot_pixel(int(x), int(y), color)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        self.plot_pixel(int(x), int(y), color)




    def draw_axes(self):
        """Draw coordinate axes with grid lines."""
        axis_color = (1.0, 1.0, 1.0)  # White for axes
        grid_color = (0.5, 0.5, 0.5)  # Gray for grid

        # Draw main X and Y axes
        self.draw_line(50, 50, 50, self.height - 50, axis_color)  # Y-axis
        self.draw_line(50, 50, self.width - 50, 50, axis_color)  # X-axis

        # Draw gridlines
        spacing = 50  # Adjust spacing between grid lines
        for i in range(50, self.width - 50, spacing):  # Vertical gridlines
            self.draw_line(i, 50, i, self.height - 50, grid_color)
        for i in range(50, self.height - 50, spacing):  # Horizontal gridlines
            self.draw_line(50, i, self.width - 50, i, grid_color)

    def draw(self):
        """Draw the graph."""
        if not self.frequencies:
            return

        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        # Draw axes
        self.draw_axes()

        # Calculate scaling factors
        max_freq = max(self.frequencies)
        spacing = (self.width - 100) / len(self.frequencies)
        height_scale = (self.height - 100) / max_freq

        # Choose algorithm
        line_algo = self.dda_line if self.algorithm == 'dda' else self.bresenham_line

        # Draw the points and lines
        x = 50
        prev_x = x
        prev_y = 50

        for i, freq in enumerate(self.frequencies):
            height = 50 + freq * height_scale

            # Draw connecting lines
            line_algo(prev_x, prev_y, x + spacing, height, (0.0, 0.0, 0.0))  # Black line

            # Mark points
            
            self.plot_pixel(x + spacing, height, (0.0, 0.0, 1.0),18.0)  # Blue points

            # Update previous points
            prev_x = x + spacing
            prev_y = height
            x += spacing

        glFlush()


def main():
    if not glfwInit():
        print("Failed to initialize GLFW")
        return

    # Create window
    window = glfwCreateWindow(800, 600, "Line Drawing Algorithm", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfwTerminate()
        return

    glfwMakeContextCurrent(window)

    # Create LineDrawing instance
    drawer = LineDrawing()

    # Get algorithm choice
    algorithm = input("Choose algorithm (1 for DDA, 2 for Bresenham): ").strip()
    drawer.algorithm = 'dda' if algorithm == '1' else 'bresenham'

    # Get frequencies
    freq_input = input("Enter frequencies (comma-separated values): ").strip()
    drawer.frequencies = [float(x.strip()) for x in freq_input.split(',')]

  
    print("Drawing graph")

    # OpenGL setup
    glClearColor(1.0, 1.0, 1.0, 1.0)  # White background
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, 800, 0, 600)
    glMatrixMode(GL_MODELVIEW)

    while not glfwWindowShouldClose(window):
        drawer.draw()
        glfwSwapBuffers(window)
        glfwPollEvents()

    glfwTerminate()


if __name__ == "__main__":
    main()
