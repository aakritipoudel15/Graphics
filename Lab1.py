from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from math import *
from glfw.GLFW import *

def draw_circle(x, y, r, num_segments):
    glBegin(GL_TRIANGLE_STRIP)
    glColor3f(0.75, 0, 0)
    for i in range(num_segments + 1):
        theta = 2.0 * 3.1415926 * i / num_segments
        outer_x = x + r * cos(theta)
        outer_y = y + r * sin(theta)
        inner_x = x + (r - 0.1) * cos(theta)
        inner_y = y + (r - 0.1) * sin(theta)
        glVertex2f(outer_x, outer_y)
        glVertex2f(inner_x, inner_y)
    glEnd()

def draw():
    glClearColor(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glBegin(GL_TRIANGLE_STRIP)
    glColor3f(0.75, 0, 0)
    glVertex2f(-0.2,0)
    glVertex2f(-0.2,0.4)
    glVertex2f(-0.1,0)
    glVertex2f(-0.1,0.4)
    glEnd()

    glBegin(GL_TRIANGLE_STRIP)
    glColor3f(0.75, 0, 0)
    glVertex2f(0.1, 0)
    glVertex2f(0.1, 0.4)
    glVertex2f(0.2, 0)
    glVertex2f(0.2, 0.4)
    glEnd()

    glBegin(GL_TRIANGLE_STRIP)
    glColor3f(0.75, 0, 0)
    glVertex2f(0.1, 0)
    glVertex2f(-0.2, 0.4)
    glVertex2f(0.2, 0)
    glVertex2f(-0.1, 0.4)
    glEnd()

    glBegin(GL_TRIANGLE_STRIP)
    glColor3f(0.75, 0, 0)
    glVertex2f(0.445, 0.135)
    glVertex2f(-0.445, 0.135)
    glVertex2f(0.445, 0.265)
    glVertex2f(-0.445, 0.265)
    glEnd()
    glFlush()


    CENTER  = (0,0.2) #According to previous renders (0.14+0.26)/2 = 0.2
    draw_circle(*CENTER, 0.45, 100)


    #draw a rectangle made up of two triangles just above the right half of the circle
    glBegin(GL_TRIANGLE_STRIP)
    glColor3f(1, 1, 1)
    glVertex2f(0.45,0.26)
    glVertex2f(0.2,0.26)
    glVertex2f(0.45,0.38)
    glVertex2f(0.2,0.38)
    glEnd()

    glBegin(GL_TRIANGLE_STRIP)
    glColor3f(1, 1, 1)
    glVertex2f(-0.45, 0.14)
    glVertex2f(-0.2, 0.14)
    glVertex2f(-0.45, 0.02)
    glVertex2f(-0.2, 0.02)
    glEnd()
 
def main():
    if not glfwInit():
        return
    window = glfwCreateWindow(600, 600, "Tourism Board", None, None)
    if not window:
        glfwTerminate()
        return
    
    

    glfwMakeContextCurrent(window)
    gluOrtho2D(-1, 1, -1, 1)
    glfwSwapInterval(1)
    while not glfwWindowShouldClose(window):
        glfwPollEvents()
        draw()
        glfwSwapBuffers(window)
    glfwTerminate()


if __name__ == "__main__":
     
    glfwInit()
    displayInfo = glfwGetVideoMode(glfwGetPrimaryMonitor())
    print(displayInfo)
        
    main()

