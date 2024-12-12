from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
from math import cos, sin, pi

# Mid-Point Circle Drawing Algorithm
def midpoint_circle_drawing_algorithm(x_center, y_center, radius):
    points = []
    x = radius
    y = 0
    p = 1 - radius

    while x > y:
        points.append((x_center + x, y_center + y))
        points.append((x_center - x, y_center + y))
        points.append((x_center + x, y_center - y))
        points.append((x_center - x, y_center - y))
        points.append((x_center + y, y_center + x))
        points.append((x_center - y, y_center + x))
        points.append((x_center + y, y_center - x))
        points.append((x_center - y, y_center - x))

        y += 1
        if p <= 0:
            p = p + 2 * y + 1
        else:
            x -= 1
            p = p + 2 * y - 2 * x + 1

    return points

def draw_midpoint_circle():
    glPointSize(4.0)  # Set the pixel size to 4*4

    points = midpoint_circle_drawing_algorithm(0, 0, 50)
    glBegin(GL_POINTS)
    for point in points:
        glVertex2f(point[0] / 100.0, point[1] / 100.0)
    glEnd()

# Mid-Point Ellipse Drawing Algorithm
def midpoint_ellipse_drawing_algorithm(x_center, y_center, rx, ry):
    points = []
    x = 0
    y = ry
    p1 = (ry**2) - (rx**2 * ry) + (0.25 * rx**2)

    while (2 * ry**2 * x) < (2 * rx**2 * y):
        points.append((x_center + x, y_center + y))
        points.append((x_center - x, y_center + y))
        points.append((x_center + x, y_center - y))
        points.append((x_center - x, y_center - y))

        if p1 < 0:
            x += 1
            p1 += 2 * ry**2 * x + ry**2
        else:
            x += 1
            y -= 1
            p1 += 2 * ry**2 * x - 2 * rx**2 * y + ry**2

    p2 = (ry**2) * (x + 0.5)**2 + (rx**2) * (y - 1)**2 - (rx**2 * ry**2)

    while y >= 0:
        points.append((x_center + x, y_center + y))
        points.append((x_center - x, y_center + y))
        points.append((x_center + x, y_center - y))
        points.append((x_center - x, y_center - y))

        if p2 > 0:
            y -= 1
            p2 -= 2 * rx**2 * y + rx**2
        else:
            y -= 1
            x += 1
            p2 += 2 * ry**2 * x - 2 * rx**2 * y + rx**2

    return points

def draw_midpoint_ellipse():
    glPointSize(4.0)  # Set the pixel size to 4*4
    points = midpoint_ellipse_drawing_algorithm(0, 0, 50, 30)
    glBegin(GL_POINTS)
    for point in points:
        glVertex2f(point[0] / 100.0, point[1] / 100.0)
    glEnd()

# Polar Coordinate Circle Drawing Algorithm
def polar_circle_drawing_algorithm(x_center, y_center, radius):
    glPointSize(4.0)  # Set the pixel size to 4*4
    points = []
    theta = 0.0
    step = 2 * pi / 360

    while theta <= 2 * pi:
        x = x_center + radius * cos(theta)
        y = y_center + radius * sin(theta)
        points.append((x, y))
        theta += step

    return points

def draw_polar_circle():
    points = polar_circle_drawing_algorithm(0, 0, 50)
    glBegin(GL_POINTS)
    for point in points:
        glVertex2f(point[0] / 100.0, point[1] / 100.0)
    glEnd()

# Drawing based on user choice
def draw(choice):
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(1.0, 0.0, 0.0)  # Red for all shapes

    if choice == 1:
        draw_midpoint_circle()
    elif choice == 2:
        draw_midpoint_ellipse()
    elif choice == 3:
        draw_polar_circle()

    glFlush()

def main():
    if not glfw.init():
        return

    window = glfw.create_window(600, 600, "Drawing Algorithms", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    gluOrtho2D(-1, 1, -1, 1)

    # Get user choice
    print("Choose a drawing algorithm:")
    print("1. Midpoint Circle")
    print("2. Midpoint Ellipse")
    print("3. Polar Circle")
    choice = int(input("Enter your choice (1/2/3): "))

    while not glfw.window_should_close(window):
        glfw.poll_events()
        draw(choice)
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
