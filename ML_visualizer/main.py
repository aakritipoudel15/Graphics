import glfw
import OpenGL.GL as gl
from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
from visualization.linear_vis import main as linear_vis
from visualization.logistic_vis import main as logistic_vis
from visualization.linear import main as linear
from visualization.logistic import main as logistic

class Button:
    def __init__(self, x, y, width, height, text, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.hovered = False

    def is_hovered(self, cursor_x, cursor_y):
        return (self.x <= cursor_x <= self.x + self.width and
                self.y <= cursor_y <= self.y + self.height)

    def draw(self):
        if self.hovered:
            gl.glColor3f(min(self.color[0] + 0.2, 1.0), 
                         min(self.color[1] + 0.2, 1.0), 
                         min(self.color[2] + 0.2, 1.0))  # Lighter when hovered
        else:
            gl.glColor3f(*self.color)  # Normal color

        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(self.x, self.y)
        gl.glVertex2f(self.x + self.width, self.y)
        gl.glVertex2f(self.x + self.width, self.y + self.height)
        gl.glVertex2f(self.x, self.y + self.height)
        gl.glEnd()

        # Button border
        gl.glColor3f(0.8, 0.8, 0.8)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex2f(self.x, self.y)
        gl.glVertex2f(self.x + self.width, self.y)
        gl.glVertex2f(self.x + self.width, self.y + self.height)
        gl.glVertex2f(self.x, self.y + self.height)
        gl.glEnd()

class MenuSystem:
    def __init__(self):
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")

        self.window = glfw.create_window(800, 600, "ML Algorithm Visualizer", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glutInit()

        # Set callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_position_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)

        self.current_menu = "main"
        self.selected_algorithm = None
        self._setup_buttons()

    def _setup_buttons(self):
        self.main_menu_buttons = [
            Button(-0.6, 0.2, 1.2, 0.15, "Linear Regression", (0.2, 0.6, 1.0)),
            Button(-0.6, -0.2, 1.2, 0.15, "Logistic Regression", (1.0, 0.4, 0.2))
        ]

        self.sub_menu_buttons = [
            Button(-0.6, 0.2, 1.2, 0.15, "Use default CSV file", (0.2, 0.8, 0.4)),
            Button(-0.6, -0.2, 1.2, 0.15, "Give Manual input through window", (0.9, 0.7, 0.2)),
            Button(-0.6, -0.6, 1.2, 0.15, "Back to Main Menu", (0.6, 0.2, 0.8))
        ]

    def render_text(self, text, x, y):
        gl.glColor3f(0.0, 0.0, 0.0)  # Black text for better visibility
        gl.glRasterPos2f(x, y)
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

    def _cursor_position_callback(self, window, xpos, ypos):
        width, height = glfw.get_window_size(window)
        x = (2.0 * xpos / width) - 1.0
        y = 1.0 - (2.0 * ypos / height)

        buttons = (self.main_menu_buttons if self.current_menu == "main" 
                  else self.sub_menu_buttons)
        
        for button in buttons:
            button.hovered = button.is_hovered(x, y)

    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(window)
            width, height = glfw.get_window_size(window)
            x = (2.0 * xpos / width) - 1.0
            y = 1.0 - (2.0 * ypos / height)

            if self.current_menu == "main":
                self._handle_main_menu_click(x, y)
            else:
                self._handle_sub_menu_click(x, y)

    def _handle_main_menu_click(self, x, y):
        for i, button in enumerate(self.main_menu_buttons):
            if button.is_hovered(x, y):
                self.selected_algorithm = "linear" if i == 0 else "logistic"
                self.current_menu = "sub"
                break

    def _handle_sub_menu_click(self, x, y):
        for i, button in enumerate(self.sub_menu_buttons):
            if button.is_hovered(x, y):
                if i == 2:  # Back button
                    self.current_menu = "main"
                    self.selected_algorithm = None
                else:
                    glfw.destroy_window(self.window)
                    if self.selected_algorithm == "linear":
                        if i == 0:
                            linear()
                        else:
                            linear_vis()
                    else:
                        if i == 0:
                            logistic()
                        else:
                            logistic_vis()
                break

    def run(self):
        gl.glClearColor(0.9, 0.9, 0.9, 1.0)  # Light background
        while not glfw.window_should_close(self.window):
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glLoadIdentity()

            if self.current_menu == "main":
                self.render_text("Select Algorithm to Visualize:", -0.4, 0.6)
                for button in self.main_menu_buttons:
                    button.draw()
                    self.render_text(button.text, button.x + 0.1, button.y + 0.05)
            else:
                self.render_text("Select Visualization Method:", -0.4, 0.6)
                for button in self.sub_menu_buttons:
                    button.draw()
                    self.render_text(button.text, button.x + 0.1, button.y + 0.05)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

def main():
    try:
        menu = MenuSystem()
        menu.run()
    except Exception as e:
        print(f"Error: {e}")
        glfw.terminate()

if __name__ == "__main__":
    main()
