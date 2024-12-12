#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Window
const int WIDTH = 800;
const int HEIGHT = 600;
static GLFWwindow* window;

// Simulation Parameters
float timeScale = 4.0f;				 // Determines simulation speed
float gravityAcc = 15.0f;			 // How fast particles acc. downward

int initParticleNumber = 800;		 // The number of particles to spawn
float pointSize = 15.0f;			 // The size of the rendered particle

float borderDampen = 0.5f;			 // Dampen border collisions
float interactionRadius = 0.2f;	 	 // Radius for neighbor interactions
float stiffness = 50.0f;        	 // Pressure constant
float restDensity = 15.0f;      	 // Rest density of the fluid
float viscosity = 25.0f;         	 // Viscosity constant
float velocityCap = 10.0f;			 // Cap velocity to stop explosions!

float mousePower = 1.0f;			 // Mouse influence over particles
float mouseInteractionRadius = 0.2f; // Radius for neighbor interactions
float borderMoveSpeed = 0.0f;		 // How fast the user can move bonudaries
float borderWidth = 1.95f;			 // Starting border width
float borderHeight = 1.95f;			 // Starting border height

// Functions
int setupGui();
int cleanupGui();
void drawGui();
void setup3D();
void moveCamera(float deltaTime);
float randomFloat(float range);

// Camera
float cameraDistance = 5.0f; // Distance from the center
float cameraSpeed = 0.5f;    // Speed of camera movement
float cameraHeight = 1.0f;   // Height of the camera

struct Camera {
	glm::vec3 position;
	glm::vec3 rotation;
};

Camera camera = {
	glm::vec3(0.0f, 2.0f, cameraDistance), // Initial position
	glm::vec3(0.0f, 0.0f, 0.0f)            // Rotation
};

// Particles
struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    float density = 0.0f;
    float pressure = 0.0f;
};

std::vector<Particle> particles;

// Spawn in the particles
void initParticles() {
    particles.clear();

    int gridSize = static_cast<int>(cbrt(initParticleNumber));
    float totalWidth = interactionRadius * (gridSize - 1);
    float startX = -totalWidth / 2.0f;
    float startY = -totalWidth / 2.0f;
    float startZ = -totalWidth / 2.0f;

    glm::vec3 velocity(0.0f);
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            for (int k = 0; k < gridSize; ++k) {
                glm::vec3 position = glm::vec3(
                    startX + i * interactionRadius + randomFloat(0.01f),
                    startY + j * interactionRadius + randomFloat(0.01f),
                    startZ + k * interactionRadius + randomFloat(0.01f)
                );
                particles.push_back({position, velocity});
            }
        }
    }
}

// Adjust particle characteristics
void computeDensityAndPressure() {
    for (auto& p : particles) {
        p.density = 0.0f;

        for (const auto& neighbor : particles) {
            glm::vec3 diff = p.position - neighbor.position;
            float distance = glm::length(diff);
            if (distance < interactionRadius) {
                float q = distance / interactionRadius;
                p.density += (1.0f - q) * (1.0f - q) * (1.0f - q); // Poly6 kernel
            }
        }

        p.pressure = stiffness * (p.density - restDensity);
    }
}

// Find the forces acting on a particle
void computeForces(float deltaTime) {
    for (auto& p : particles) {
        glm::vec3 pressureForce(0.0f);
        glm::vec3 viscosityForce(0.0f);

        for (const auto& neighbor : particles) {
            glm::vec3 diff = p.position - neighbor.position;
            float distance = glm::length(diff);

            if (distance < interactionRadius && distance > 0.0f) {
                glm::vec3 direction = glm::normalize(diff);

                float q = distance / interactionRadius;
                float pressureKernel = (1.0f - q);
                pressureForce -= direction * (p.pressure + neighbor.pressure) / (2.0f * neighbor.density) * pressureKernel;
                viscosityForce += viscosity * (neighbor.velocity - p.velocity) * pressureKernel;
            }
        }

        glm::vec3 gravity(0.0f, -gravityAcc, 0.0f);
        p.velocity += (pressureForce + viscosityForce + gravity) * deltaTime;

        if (glm::length(p.velocity) > velocityCap) {
            p.velocity = glm::normalize(p.velocity) * velocityCap;
        }
    }
}

void updateParticles(float deltaTime) {
    computeDensityAndPressure();
    computeForces(deltaTime);

    float left = -borderWidth / 2.0f;
    float right = borderWidth / 2.0f;
    float top = borderHeight / 2.0f;
    float bottom = -borderHeight / 2.0f;
    float front = -borderWidth / 2.0f;
    float back = borderWidth / 2.0f;

    for (auto& p : particles) {
        p.position += p.velocity * deltaTime;

        // Boundary conditions (x-axis)
        if (p.position.x < left) {
            p.position.x = left;
            p.velocity.x *= -borderDampen;
        }
        if (p.position.x > right) {
            p.position.x = right;
            p.velocity.x *= -borderDampen;
        }

        // Boundary conditions (y-axis)
        if (p.position.y < bottom) {
            p.position.y = bottom;
            p.velocity.y *= -borderDampen;
        }
        if (p.position.y > top) {
            p.position.y = top;
            p.velocity.y *= -borderDampen;
        }

        // Boundary conditions (z-axis)
        if (p.position.z < front) {
            p.position.z = front;
            p.velocity.z *= -borderDampen;
        }
        if (p.position.z > back) {
            p.position.z = back;
            p.velocity.z *= -borderDampen;
        }
    }
}

// Draw the particles to the screen as points
void renderParticles() {
	glEnable(GL_POINT_SMOOTH); // Enable smooth points
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST); // Use the nicest smoothing algorithm
    glPointSize(pointSize);
    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        float normalizedDensity = glm::clamp((p.density - restDensity) / restDensity, 0.0f, 1.0f);
        float r = normalizedDensity;
        float g = 0.0f;
        float b = 1.0f - normalizedDensity;

        glColor3f(r, g, b);
        glVertex3f(p.position.x, p.position.y, p.position.z);
    }
    glEnd();
}

float randomFloat(float range) {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2.0f * range - range;
}

void drawBorder() {
    // Define the borders of the cube
    float left = -borderWidth / 2.0f;
    float right = borderWidth / 2.0f;
    float top = borderHeight / 2.0f;
    float bottom = -borderHeight / 2.0f;
    float front = -borderWidth / 2.0f;
    float back = borderWidth / 2.0f;

    // Set the color for the border
    glColor3f(0.5f, 0.5f, 0.5f); // Gray color for the border
    glLineWidth(2.0f);           // Set line width

    // Draw the edges of the cube
    glBegin(GL_LINES);

    // Bottom face
    glVertex3f(left, bottom, front);
    glVertex3f(right, bottom, front);

    glVertex3f(right, bottom, front);
    glVertex3f(right, bottom, back);

    glVertex3f(right, bottom, back);
    glVertex3f(left, bottom, back);

    glVertex3f(left, bottom, back);
    glVertex3f(left, bottom, front);

    // Top face
    glVertex3f(left, top, front);
    glVertex3f(right, top, front);

    glVertex3f(right, top, front);
    glVertex3f(right, top, back);

    glVertex3f(right, top, back);
    glVertex3f(left, top, back);

    glVertex3f(left, top, back);
    glVertex3f(left, top, front);

    // Vertical edges
    glVertex3f(left, bottom, front);
    glVertex3f(left, top, front);

    glVertex3f(right, bottom, front);
    glVertex3f(right, top, front);

    glVertex3f(right, bottom, back);
    glVertex3f(right, top, back);

    glVertex3f(left, bottom, back);
    glVertex3f(left, top, back);

    glEnd();
}

void processInput() {
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        borderHeight += borderMoveSpeed; // Increase height
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        borderHeight -= borderMoveSpeed; // Decrease height
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        borderWidth += borderMoveSpeed; // Increase width
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        borderWidth -= borderMoveSpeed; // Decrease width
    }

    // Minimum size for the border
    borderWidth = std::max(0.1f, borderWidth);
    borderHeight = std::max(0.1f, borderHeight);
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main() {
	std::srand(std::time(nullptr));

    setupGui();
	setup3D();
    initParticles();

    float lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime();
        float deltaTime = (currentTime - lastTime) / timeScale;
        lastTime = currentTime;

		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

        processInput();

        glClear(GL_COLOR_BUFFER_BIT);

		moveCamera(deltaTime);
        drawBorder();
        updateParticles(deltaTime);
        renderParticles();
        drawGui();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cleanupGui();
    return 0;
}

int setupGui() {
    glEnable(GL_PROGRAM_POINT_SIZE);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }

    // Init OpenGL + window
    window = glfwCreateWindow(WIDTH, HEIGHT, "opengl", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create window!" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    // glfwSwapInterval(1);

    // Init ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void) io;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();

    return 0;
}

// Cleanup
int cleanupGui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

void drawGui() {
    // Create Frame + Fullscreen
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Make window
    ImGui::Begin("Fluid Sim Parameters", NULL, ImGuiWindowFlags_AlwaysAutoResize);
	ImGui::Text("Framerate: %.1f FPS", ImGui::GetIO().Framerate);
	ImGui::InputInt("Number", &initParticleNumber);
	ImGui::InputFloat("Spacing", &interactionRadius); // Corrected missing semicolon
	ImGui::SliderFloat("Size", &pointSize, 1.0f, 100.0f);
	if (ImGui::Button("Start Simulation"))
		initParticles();
	if (ImGui::CollapsingHeader("Parameters")) {
		ImGui::SliderFloat("Time Scale", &timeScale, 1.0f, 10.0f);
		ImGui::InputFloat("Gravity", &gravityAcc);
		ImGui::InputFloat("Border Dampening", &borderDampen);
		ImGui::Separator();
		ImGui::InputFloat("Interact Radius", &interactionRadius);
		ImGui::InputFloat("Stiffness", &stiffness);
		ImGui::InputFloat("Rest Density", &restDensity);
		ImGui::InputFloat("Viscosity", &viscosity);
		ImGui::Separator();
		ImGui::InputFloat("Max Velocity", &velocityCap);
	}
	if (ImGui::CollapsingHeader("Camera")) {
		ImGui::SliderFloat("Cam Speed", &cameraSpeed, 0.0f, 10.0f);
		ImGui::SliderFloat("Cam Distance", &cameraDistance, 0.1f, 10.0f);
		ImGui::SliderFloat("Cam Height", &cameraHeight, 0.0f, 5.0f);
	}
	if (ImGui::CollapsingHeader("User Interaction")) {
		ImGui::SliderFloat("Border Move Speed", &borderMoveSpeed, 0.0f, 0.1f);
	}
    ImGui::End();

    // Render GUI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void setup3D() {
    // Projection matrix
    glm::mat4 projection = glm::perspective(
        glm::radians(cameraFov),
        (float)WIDTH / (float)HEIGHT,
        0.1f, 100.0f
    );

    // View matrix (camera)
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 5.0f), // Camera position
        glm::vec3(0.0f, 0.0f, 0.0f), // Look-at point
        glm::vec3(0.0f, 1.0f, 0.0f)  // Up vector
    );

    // Set the matrices in OpenGL
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(glm::value_ptr(projection));
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(glm::value_ptr(view));
}

void moveCamera(float deltaTime) {
    static float angle = 0.0f;
    angle += cameraSpeed * deltaTime;

    // Calculate the new position
    camera.position.x = cameraDistance * cos(angle);
    camera.position.z = cameraDistance * sin(angle);
    camera.position.y = cameraHeight;

    glm::vec3 target(0.0f, 0.0f, 0.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);

    // Load the view matrix into OpenGL
    glm::mat4 view = glm::lookAt(camera.position, target, up);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(glm::value_ptr(view));
}
