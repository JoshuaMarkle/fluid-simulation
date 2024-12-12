#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <cmath>

const int WIDTH = 800;
const int HEIGHT = 600;
static GLFWwindow* window;

float timeScale = 4.0f;
float gravityAcc = 15.0f;

float initSpacing = 0.05f;
int initParticleNumber = 400;
float pointSize = 30.0f;

float borderDampen = 0.5f;		 // Dampen border collisions
float interactionRadius = 0.05f; // Radius for neighbor interactions
float stiffness = 100.0f;        // Pressure constant
float restDensity = 10.0f;       // Rest density of the fluid
float viscosity = 1.0f;          // Viscosity constant
float velocityCap = 10.0f;		 // Cap velocity to stop explosions!

float mousePower = 1.0f;
float mouseInteractionRadius = 0.2f; // Radius for neighbor interactions
float borderMoveSpeed = 0.0f;

float borderWidth = 1.95f;
float borderHeight = 1.95f;

int setupGui();
int cleanupGui();
void drawGui();
glm::vec2 processMouseInteraction(float deltaTime);
void renderMouseInteraction(const glm::vec2& mousePosition);

struct Particle {
    glm::vec2 position;
    glm::vec2 velocity;
    float density = 0.0f;
    float pressure = 0.0f;
};

std::vector<Particle> particles;

void initParticles() {
    int gridSize = static_cast<int>(sqrt(initParticleNumber));
    float totalWidth = initSpacing * (gridSize - 1);
    float startX = -totalWidth / 2.0f; // Center horizontally
    float startY = -totalWidth / 2.0f; // Center vertically

	particles.clear();
    glm::vec2 velocity = glm::vec2(0.0f, 0.0f);
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            glm::vec2 position = glm::vec2(startX + i * initSpacing, startY + j * initSpacing);
            if (j % 2 == 1)
                position.x += initSpacing / 2.0f;
            particles.push_back({position, velocity});
        }
    }
}

void computeDensityAndPressure() {
    for (auto& p : particles) {
        p.density = 0.0f;

        for (const auto& neighbor : particles) {
            glm::vec2 diff = p.position - neighbor.position;
            float distance = glm::length(diff);

            if (distance < interactionRadius) {
                // Kernel function: Poly6 kernel
                float q = distance / interactionRadius;
                p.density += (1.0f - q) * (1.0f - q) * (1.0f - q); // Simplified
            }
        }

        // Pressure: P = stiffness * (density - rest density)
        p.pressure = stiffness * (p.density - restDensity);
    }
}

void computeForces(float deltaTime) {
    for (auto& p : particles) {
        glm::vec2 pressureForce(0.0f);
        glm::vec2 viscosityForce(0.0f);

        for (const auto& neighbor : particles) {
            glm::vec2 diff = p.position - neighbor.position;
            float distance = glm::length(diff);

            if (distance < interactionRadius && distance > 0.0f) {
                glm::vec2 direction = glm::normalize(diff);

                // Pressure force: F = -grad(P) * kernel
                float q = distance / interactionRadius;
                float pressureKernel = (1.0f - q);
                pressureForce -= direction * (p.pressure + neighbor.pressure) / (2.0f * neighbor.density) * pressureKernel;

                // Viscosity force: F = mu * laplacian(v) * kernel
                viscosityForce += viscosity * (neighbor.velocity - p.velocity) * pressureKernel;
            }
        }

        // Combine forces
        glm::vec2 gravity = glm::vec2(0.0f, -gravityAcc);
        p.velocity += (pressureForce + viscosityForce + gravity) * deltaTime;

        // Cap velocity magnitude (to stop random particle explosions)
		if (glm::length(p.velocity) > velocityCap)
			p.velocity = glm::normalize(p.velocity) * velocityCap;
    }
}

void correctPositions() {
    for (auto& p : particles) {
        glm::vec2 correction(0.0f);
        for (const auto& neighbor : particles) {
            glm::vec2 diff = p.position - neighbor.position;
            float distance = glm::length(diff);

            if (distance < interactionRadius && distance > 0.0f) {
                correction += (1.0f - distance / interactionRadius) * 0.01f * diff;
            }
        }
        p.position += correction;
    }
}

void updateParticles(float deltaTime) {
    computeDensityAndPressure();
    computeForces(deltaTime);

    // Boundary conditions
    float left = -borderWidth / 2.0f;
    float right = borderWidth / 2.0f;
    float top = borderHeight / 2.0f;
    float bottom = -borderHeight / 2.0f;

    for (auto& p : particles) {
        // Update position
        p.position += p.velocity * deltaTime;

        // Boundary conditions
        if (p.position.x < left) {
            p.position.x = left; // Clamp to the left boundary
            p.velocity.x *= -borderDampen; // Reverse and dampen velocity
        }
        if (p.position.x > right) {
            p.position.x = right; // Clamp to the right boundary
            p.velocity.x *= -borderDampen; // Reverse and dampen velocity
        }
        if (p.position.y < bottom) {
            p.position.y = bottom; // Clamp to the bottom boundary
            p.velocity.y *= -borderDampen; // Reverse and dampen velocity
        }
        if (p.position.y > top) {
            p.position.y = top; // Clamp to the top boundary
            p.velocity.y *= -borderDampen; // Reverse and dampen velocity
        }
    }

    correctPositions(); // Re-enabled position correction
}

void renderParticles() {
    glEnable(GL_POINT_SMOOTH); // Enable smooth points
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST); // Use the nicest smoothing algorithm
    glPointSize(pointSize); // Set the size of the points (adjust for visibility)

    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        // Set color based on density (red for high density, blue for low)
        float normalizedDensity = glm::clamp((p.density - restDensity) / restDensity, 0.0f, 1.0f);
        float r = normalizedDensity;      // Red increases with density
        float g = 0.0f;                   // No green in the gradient
        float b = 1.0f - normalizedDensity; // Blue decreases with density

        glColor3f(r, g, b); // Apply the color
        glVertex2f(p.position.x, p.position.y); // Draw the particle
    }
    glEnd();
}

void drawBorder() {
    float left = -borderWidth / 2.0f;
    float right = borderWidth / 2.0f;
    float top = borderHeight / 2.0f;
    float bottom = -borderHeight / 2.0f;

    glColor3f(0.5f, 0.5f, 0.5f); // Set border color to white
    glBegin(GL_LINE_LOOP);
    glVertex2f(left, top);    // Top-left corner
    glVertex2f(right, top);   // Top-right corner
    glVertex2f(right, bottom); // Bottom-right corner
    glVertex2f(left, bottom); // Bottom-left corner
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

    // Ensure minimum size for the border
    borderWidth = std::max(0.1f, borderWidth);
    borderHeight = std::max(0.1f, borderHeight);
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main() {

    setupGui();
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
		glm::vec2 mousePosition = processMouseInteraction(deltaTime); 

        glClear(GL_COLOR_BUFFER_BIT);

        drawBorder();
        updateParticles(deltaTime);
        renderParticles();
		renderMouseInteraction(mousePosition);
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
    ImGui::Begin("Fluid Sim Parameters");
	ImGui::Text("Framerate: %.1f FPS", ImGui::GetIO().Framerate);
	ImGui::InputInt("Number", &initParticleNumber);
	ImGui::InputFloat("Spacing", &initSpacing); // Corrected missing semicolon
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
	if (ImGui::CollapsingHeader("User Interaction")) {
		ImGui::SliderFloat("Mouse Power", &mousePower, 0.1f, 20.0f);
		ImGui::SliderFloat("Mouse Radius", &mouseInteractionRadius, 0.1f, 1.0f);
		ImGui::SliderFloat("Border Move Speed", &borderMoveSpeed, 0.0f, 0.1f);
	}
    ImGui::End();

    // Render GUI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

glm::vec2 processMouseInteraction(float deltaTime) {
    // Get mouse position
    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);

    // Convert screen coordinates to NDC
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
	float ndcX = (mouseX / display_w) * 2.0f - 1.0f;
	float ndcY = 1.0f - (mouseY / display_h) * 2.0f;

	float simX = ndcX * borderWidth + 1.0f;
	float simY = ndcY * borderHeight - 1.0f;
	glm::vec2 mousePosition(simX, simY);
	// std::cout << "Mouse Position: (" << mousePosition.x << ", " << mousePosition.y << ")\n";

    // Check for mouse buttons
    bool leftClick = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    bool rightClick = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    if (leftClick || rightClick) {
        // Apply forces to particles
        for (auto& p : particles) {
            glm::vec2 direction = p.position - mousePosition;
            float distance = glm::length(direction);

            if (distance < mouseInteractionRadius) {
                float strength = (1.0f - distance / mouseInteractionRadius) * mousePower * 100.0f;
                glm::vec2 normalizedDirection = glm::normalize(direction);
                glm::vec2 force = glm::vec2(0.0f);

                if (rightClick) {
                    force = normalizedDirection * strength; // Push away
                } else if (leftClick) {
                    force = -normalizedDirection * strength; // Pull in
                }

                p.velocity += force * deltaTime;
            }
        }
    }

    return mousePosition; // Return the computed mouse position
}

void renderMouseInteraction(const glm::vec2& mousePosition) {
    glColor3f(1.0f, 1.0f, 1.0f); // White color for the interaction circle
    glLineWidth(2.0f);

    glBegin(GL_LINE_LOOP);
    const int numSegments = 100; // Circle resolution
    for (int i = 0; i < numSegments; ++i) {
        float theta = 2.0f * M_PI * float(i) / float(numSegments); // Angle for this segment
        float x = cosf(theta) * mouseInteractionRadius;
        float y = sinf(theta) * mouseInteractionRadius;
        glVertex2f(mousePosition.x + x, mousePosition.y + y);
    }
    glEnd();
}
