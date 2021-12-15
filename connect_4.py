import numpy as np
import pygame
import sys
import math
import random

from collections import deque

#from wrapt.wrappers import transient_function_wrapper
from model import *

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
	board = np.zeros((ROW_COUNT,COLUMN_COUNT))
	return board

def drop_piece(board, row, col, piece):
	board[row][col] = piece

def is_valid_location(board, col):
	return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
	for r in range(ROW_COUNT):
		if board[r][col] == 0:
			return r

def print_board(board):
	print(np.flip(board, 0))

def winning_move(board, piece):
	# Check horizontal locations for win
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT):
			if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
				return True

	# Check vertical locations for win
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
				return True

	# Check positively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
				return True

	# Check negatively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(3, ROW_COUNT):
			if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
				return True

def draw_board(board):
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):
			pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
			pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
	
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):		
			if board[r][c] == 1:
				pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
			elif board[r][c] == -1: ## 2 
				pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
	pygame.display.update()

def restart():
	global board, game_over, turn
	board = create_board()
	game_over = False
	turn = 1
	draw_board(board)
	pygame.display.update()


board = create_board()
print_board(board)
game_over = False
turn = 0

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 75)

def play():
	while not game_over:

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()

			if event.type == pygame.MOUSEMOTION:
				pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
				posx = event.pos[0]
				if turn == 0:
					pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
				else: 
					pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
			pygame.display.update()

			if event.type == pygame.MOUSEBUTTONDOWN:
				pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
				#print(event.pos)
				# Ask for Player 1 Input
				if turn == 0:
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))

					if is_valid_location(board, col):
						row = get_next_open_row(board, col)
						drop_piece(board, row, col, 1)

						if winning_move(board, 1):
							label = myfont.render("Player 1 wins!!", 1, RED)
							screen.blit(label, (40,10))
							game_over = True


				# # Ask for Player 2 Input
				else:				
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))

					if is_valid_location(board, col):
						row = get_next_open_row(board, col)
						drop_piece(board, row, col, -1) ## 2

						if winning_move(board, -1):## 2 
							label = myfont.render("Player 2 wins!!", 1, YELLOW)
							screen.blit(label, (40,10))
							game_over = True

				#print_board(board)
				#draw_board(board)

				turn += 1
				turn = turn % 2

				if game_over:
					#pygame.time.wait(3000)
					restart()

def get_random_valid(board):
	valid = []
	for i, value in enumerate(board[5]):
		if value == 0.0:
			valid.append(i)
	return random.choice(valid)

def run(episodes=500000, batch_size=32, save_interval=500):
	global game_over, turn
	episode = 0
	batch = 0
	step = 0
	decay = 0.0005

	update_interval = 150

	epsilon = 1
	max_epsilon = 1
	min_epsilon = 0.05
	p1_started = False
	p2_started = False
	p1_prev_state = None
	p2_prev_state = None
	p1_prev_reward = 0
	p2_prev_reward = 0
	p1_col = 0 # action
	p2_col = 0

	turn = 1

	history = deque(maxlen=20000)

	for i in range(episodes):
		#winner = None
		while not game_over:
			step += 1
			state = np.array(board)#.flatten()
			reward = 0
			if turn == -1:
				state = state * np.array(-1)
				p2_prev_state = state
			else:
				p1_prev_state = state
			
			random_num = np.random.rand()

			if random_num <= epsilon:
				#col = random.randrange(7)
				col = get_random_valid(board)
			else: col = get_move(state, turn)

			if is_valid_location(board, col):
				#reward += 1
				row = get_next_open_row(board, col)
				drop_piece(board, row, col, turn)

				if winning_move(board, turn):
					#winner = turn
					game_over = True
					#print("winner")
					reward += 500
					#print("won")
			else: 
				#winner = -1
				#game_over = True
				game_over = True
				reward -= 50
				#print("illegal move")

			# state, action, next_state, reward, done

			if turn == 1 and p1_started:
				if game_over:
					event = [state, col, state, reward, True]
				else:
					event = [p1_prev_state, p1_col, state, p1_prev_reward, False]
					p1_prev_state = state
					p1_col = col
					p1_prev_reward = reward
				history.append(event)
			elif p2_started:
				if game_over:
					event = [state, col, state, reward, True]#  state for next_state, as it wont be used either way
				else:
					event = [p2_prev_state, p2_col, state, p2_prev_reward, False]
					p2_prev_state = state
					p2_col = col
					p2_prev_reward = reward
				history.append(event)
			elif p1_started == False:
				p1_started = True
				p1_prev_state = state
				p1_col = col
				p1_prev_reward = reward
			else:
				p2_started = True
				p2_prev_state = state
				p2_col = col
				p2_prev_reward = reward



			turn *= -1
			if not game_over:
				game_over = not 0.0 in board[5]

			if step % 6 == 0 and len(history) > batch_size:
				train(history, batch_size)
			if step % update_interval == 0:
				update_working_weights()
		
		print('episode: {}'.format(episode))
		epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
		# if winner == 1:
		# 	won = True
		# else: won = False
		#update_weights(won)
		batch += 1
		episode += 1
		#if batch >= batch_size:
		#	update_working_weights()
		if episode % save_interval == 0:
			working_model.save("C:/Users/Neo/Documents/gitrepos/Connect4-Python/model.h5")
		p1_started = False
		p2_started = False
		restart()

def human_vs_nn():
	global game_over, turn
	while not game_over:

		if turn == 0:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()

				if event.type == pygame.MOUSEMOTION:
					pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
					posx = event.pos[0]
					#if turn == 0:
					pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
					#else: 
					#	pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
				pygame.display.update()

				if event.type == pygame.MOUSEBUTTONDOWN:
					pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
					#print(event.pos)
					# Ask for Player 1 Input
					#if turn == 0:
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))

					if is_valid_location(board, col):
						row = get_next_open_row(board, col)
						drop_piece(board, row, col, 1)

						if winning_move(board, 1):
							label = myfont.render("Player 1 wins!!", 1, RED)
							screen.blit(label, (40,10))
							game_over = True
					
					draw_board(board)

					turn += 1
					turn = turn % 2

				# # Ask for Player 2 Input
		else:				
			#posx = event.pos[0]
			#col = int(math.floor(posx/SQUARESIZE))
			state = np.array(board)
			col = get_move(state)


			if is_valid_location(board, col):
				row = get_next_open_row(board, col)
				drop_piece(board, row, col, -1) ## 2

				if winning_move(board, -1):## 2 
					label = myfont.render("Player 2 wins!!", 1, YELLOW)
					screen.blit(label, (40,10))
					game_over = True
			else:
				print("invalid")
				col = get_random_valid(board)
				
				row = get_next_open_row(board, col)
				drop_piece(board, row, col, -1) ## 2
				
				if winning_move(board, -1):## 2 
					label = myfont.render("Player 2 wins!!", 1, YELLOW)
					screen.blit(label, (40,10))
					game_over = True
				
			draw_board(board)

			turn += 1
			turn = turn % 2
				#print_board(board)

		if game_over:
			#pygame.time.wait(3000)
			print(board)
			restart()


run(batch_size=64)
# human_vs_nn()