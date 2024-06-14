import pygame
from Agent import NN
from copy import deepcopy
from math import hypot
import random

pygame.init()
 
# Font that is used to render the text
font20 = pygame.font.Font('freesansbold.ttf', 20)
 
# RGB values of standard colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
 
# Basic parameters of the screen
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")
 
clock = pygame.time.Clock()    
FPS = 30

RIGHT_ACTIONS = ([pygame.K_UP, pygame.K_DOWN])
LEFT_ACTIONS = ([pygame.K_w, pygame.K_s])
AGENT1_FILE = "leftAgent.npz"
AGENT2_FILE = "rightAgent.npz"
# Striker class
class Striker:
        # Take the initial position, dimensions, speed and color of the object
    def __init__(self, posx, posy, width, height, speed, color):
        self.posx = posx
        self.posy = posy
        self.width = width
        self.height = height
        self.speed = speed
     
        self.color = color
        # Rect that is used to control the position and collision of the object
        self.geekRect = pygame.Rect(posx, posy, width, height)
        # Object that is blit on the screen
        self.geek = pygame.draw.rect(screen, self.color, self.geekRect)
 
    # Used to display the object on the screen
    def display(self):
        self.geek = pygame.draw.rect(screen, self.color, self.geekRect)
 
    def update(self, yFac):
        self.posy = self.posy + self.speed*yFac
 
        # Restricting the striker to be below the top surface of the screen
        if self.posy <= 0:
            self.posy = 0
        # Restricting the striker to be above the bottom surface of the screen
        elif self.posy + self.height >= HEIGHT:
            self.posy = HEIGHT-self.height
 
        # Updating the rect with the new values
        self.geekRect = (self.posx, self.posy, self.width, self.height)
 
    def displayScore(self, text, score, x, y, color):
        text = font20.render(text+str(score), True, color)
        textRect = text.get_rect()
        textRect.center = (x, y)
 
        screen.blit(text, textRect)
 
    def getRect(self):
        return self.geekRect
    def reset(self):
        self.posy = HEIGHT/2
 
# Ball class
 
 
class Ball:
    def __init__(self, posx, posy, radius, speed, color):
        self.posx = posx
        self.posy = posy
        self.radius = radius
        self.speed = speed
        
        self.originalspeed = speed
        
        self.color = color
        self.xFac = 1
        self.yFac = -1
        self.ball = pygame.draw.circle(
            screen, self.color, (self.posx, self.posy), self.radius)
        self.firstTime = 1
 
    def display(self):
        self.ball = pygame.draw.circle(
            screen, self.color, (self.posx, self.posy), self.radius)
 
    def update(self):
        self.posx += self.speed*self.xFac
        self.posy += self.speed*self.yFac
 
        # If the ball hits the top or bottom surfaces, 
        # then the sign of yFac is changed and 
        # it results in a reflection
        if self.posy <= 0 or self.posy >= HEIGHT:
            self.yFac *= -1
 
        if self.posx <= 0 and self.firstTime:
            self.firstTime = 0
            return 1
        elif self.posx >= WIDTH and self.firstTime:
            self.firstTime = 0
            return -1
        else:
            return 0
 
    def reset(self):
        self.posx = WIDTH//2
        self.posy = HEIGHT//2
        self.yFac = random.uniform(-1, 1)
        self.xFac *= -1
        self.firstTime = 1
 
    # Used to reflect the ball along the X-axis
    def hit(self, striker):
        center = striker.posy + striker.height / 2
        relativeIntersect = (self.posy - center) / (striker.height / 2)
        self.yFac = relativeIntersect
        self.xFac *= -1

 
    def getRect(self):
        return self.ball
 
# Game Manager
 

def gameLoop():

    
    running = True
    # Defining the objects
    geek1 = Striker(20, HEIGHT/2, 10, 100, 20, GREEN) #left
    geek2 = Striker(WIDTH-30, HEIGHT/2, 10, 100, 20, GREEN) #right
    ball = Ball(WIDTH//2, HEIGHT//2, 7, 10, WHITE)
    listOfGeeks = [geek1, geek2]
 
    # Initial parameters of the players
    geek1Score, geek2Score = 0, 0
    geek1YFac, geek2YFac = 0, 0
    screenDiagonal = hypot(WIDTH, HEIGHT)
    
    def getState(agent):
        state = []
        state.append(normalize(ball.posx,WIDTH))
        state.append(normalize(ball.posy,HEIGHT))
        state.append(ball.yFac)
        if agent == "left":
            state.append(hypot(geek1.posx - ball.posx, geek1.posy - ball.posy)/screenDiagonal)  
            lefty = normalize(geek1.posy,HEIGHT)
            state.append(lefty)      
        else:
            state.append(hypot(geek2.posx - ball.posx, geek2.posy - ball.posy)/screenDiagonal)
            state.append(normalize(geek2.posy,HEIGHT))
        return state

    LeftAgent = NN([5, 24, 2])
    LeftAgent.tag = "left"
    LeftAgent.load_weights(AGENT1_FILE)
    RightAgent = NN([5, 24, 2])
    RightAgent.load_weights(AGENT2_FILE)

    # LeftAgent.epsilon = 0
    # RightAgent.epsilon = 0
    
    Agents = [LeftAgent, RightAgent]
    
    timestep = 0
    ballBug = 0
    
    previousStateRight = None
    previousStateLeft = None
    while running:
        timestep += 1
        ballBug += 1       
        
        screen.fill(BLACK)
        stateRight = getState('right')
        stateLeft = getState("left")
        
        if previousStateRight and previousStateLeft:
            memRight = (previousStateRight,actionIndexR,rewardRight,stateRight, done)
            memLeft = (previousStateLeft,actionIndexL,rewardLeft,stateLeft, done)
            RightAgent.storeMemory(memRight)
            LeftAgent.storeMemory(memLeft)
            
            if timestep == 4:
                timestep = 0
                RightAgent.updateBatch()
                LeftAgent.updateBatch()
            
        previousStateRight = deepcopy(stateRight)
        previousStateLeft = deepcopy(stateLeft)
        
        actionIndexR = RightAgent.getAction(stateRight)
        actionR = RIGHT_ACTIONS[actionIndexR]
        
        actionIndexL = LeftAgent.getAction(stateLeft)  
        actionL = LEFT_ACTIONS[actionIndexL]
        

        rewardRight = 0
        rewardLeft = 0
        
        done = False
        
        eventR = pygame.event.Event(pygame.KEYDOWN, key=actionR)
        pygame.event.post(eventR)
        eventL = pygame.event.Event(pygame.KEYDOWN, key=actionL)   
        pygame.event.post(eventL)
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                LeftAgent.save_weights(AGENT1_FILE)
                RightAgent.save_weights(AGENT2_FILE)
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    geek2YFac = -1
                if event.key == pygame.K_DOWN:
                    geek2YFac = 1
                if event.key == pygame.K_w:
                    geek1YFac = -1
                if event.key == pygame.K_s:
                    geek1YFac = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    geek2YFac = 0
                if event.key == pygame.K_w or event.key == pygame.K_s:
                    geek1YFac = 0
 
        # Collision detection
        for i in range(len(listOfGeeks)):
            if pygame.Rect.colliderect(ball.getRect(), listOfGeeks[i].getRect()):
                if ballBug > 25:
                    if i == 0:
                        rewardLeft = 1
                    else:
                        rewardRight = 1
                    ballBug = 0
                Agents[i].bounces += 1
                if Agents[i].bounces > 9:
                    Agents[i].goals += 1
                    Agents[i].bounces = 0
                ball.hit(listOfGeeks[i])
 
        ballY = normalize(ball.posy, HEIGHT)
        ballX = normalize(ball.posx, WIDTH)
        
        # this block of code rewards the agent for following the ball with the board
        
        if ballX < 0 and ball.xFac == -1:
            agentY = normalize(geek1.posy, HEIGHT)
            if abs(ballY - agentY) < 0.25:
                rewardLeft = 0.05
            elif (ballY - agentY) > 0.25 and actionIndexL == 0:
                rewardLeft = -0.05
            elif (ballY - agentY) < -0.25 and actionIndexL == 1:
                rewardLeft = -0.05
        
        if ballX > 0 and ball.xFac == 1:   
            agentY = normalize(geek2.posy, HEIGHT)
            if abs(ballY - agentY) < 0.25: #if (ballY - agentY) < -0.5:
                rewardRight = 0.05
            elif (ballY - agentY) > 0.25 and actionIndexR == 0:
                rewardRight = -0.05
            elif (ballY - agentY) < -0.25 and actionIndexR == 1:
                rewardRight = -0.05
                

        # this block of code rewards the agent for moving in the direction of the ball
        
        # if (ballY - agentY) < -0.5:  
        #     if actionIndexR == 0:
        #         rewardRight = 0.1
        #     else:
        #         rewardRight = -0.1 #-0.15 +0.15*(1 - exp(-3*(abs(ballX - agentX))))
                
        # elif (ballY - agentY) > 0.5:
        #     if actionIndexR == 1:
        #         rewardRight = 0.1
        #     else: 
        #         rewardRight = -0.1 #-0.15 +0.15*(1 - exp(-3*(abs(ballX - agentX))))
                   
        
        # Updating the objects
        geek1.update(geek1YFac)
        geek2.update(geek2YFac)
        point = ball.update()

        # -1 -> Geek_1 has scored
        # +1 -> Geek_2 has scored
        #  0 -> None of them scored
        if point == -1:
            geek1Score += 1
            rewardRight = -1
            if Agents[0].bounces > 0:
                Agents[0].goals += 1
                Agents[0].epsilon = Agents[0].epsilon * 0.995
                Agents[0].bounces = 0
            done = True
            
        elif point == 1:
            geek2Score += 1
            rewardLeft = -1
            if Agents[1].bounces > 0:
                Agents[1].goals += 1
                Agents[1].epsilon = Agents[1].epsilon * 0.995
                Agents[1].bounces = 0
            done = True
            
        
        # Someone has scored
        # a point and the ball is out of bounds.
        # So, we reset it's position
        if point:   
            ball.reset()
            geek1.reset()
            geek2.reset()
            ball.speed = ball.originalspeed

        # Displaying the objects on the screen
        geek1.display()
        geek2.display()
        ball.display()
 
        # Displaying the scores of the players
        geek1.displayScore("Geek_1 : ", 
                           geek1Score, 100, 20, WHITE)
        geek2.displayScore("Geek_2 : ", 
                           geek2Score, WIDTH-100, 20, WHITE)
 
        pygame.display.update()
        clock.tick(FPS)     
 
# function to normalize a value to a range of -1 to 1
def normalize(n, maxValue):
    return n/(maxValue/2) - 1
 
pygame.K_w
if __name__ == "__main__":
    gameLoop()
    pygame.quit()