import numpy as np
import cv2
import random

def compress_image(image, p, q):
    height, width = image.shape
    compressed_size = 0

    for i in range(0, height, p):
        for j in range(0, width, q):
            block = image[i:i+p, j:j+q]
            unique, counts = np.unique(block, return_counts=True)
            if len(unique) == 1:
                compressed_size += 1
            else:
                compressed_size += p * q
    
    return compressed_size

def calc_compress(original_size, compressed_size):
    return original_size / compressed_size

def calc_data_red(cr):
    return 1 - (1 / cr)

def initialize_population(pop_size, dimensions):
    population = []
    for _ in range(pop_size):
        p = random.randint(1, dimensions)
        q = random.randint(1, dimensions)
        population.append((p, q))
    return population

def evaluate_population(population, image, original_size):
    fitness_scores = []
    for individual in population:
        p, q = individual
        compressed_size = compress_image(image, p, q)
        cr = calc_compress(original_size, compressed_size)
        fitness_scores.append(cr)
    return fitness_scores

def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [f / total_fitness for f in fitness_scores]
    parents = random.choices(population, weights=selection_probs, k=len(population))
    return parents

def crossover(parents):
    crossover_rate = 0.8
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1] if i+1 < len(parents) else parents[0]
        
        if random.random() < crossover_rate:
            cross_point = random.randint(1, len(parent1)-1)
            child1 = parent1[:cross_point] + parent2[cross_point:]
            child2 = parent2[:cross_point] + parent1[cross_point:]
        else:
            child1, child2 = parent1, parent2
        
        offspring.extend([child1, child2])
    return offspring

def mutate(offspring, mutation_rate=0.1, max_dim=100):
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            p = random.randint(1, max_dim)
            q = random.randint(1, max_dim)
            offspring[i] = (p, q)
    return offspring

def optimize_block_size(image, pop_size=20, generations=50, max_dim=100):
    height, width = image.shape
    original_size = height * width
    
    population = initialize_population(pop_size, max_dim)
    
    for gen in range(generations):
        fitness_scores = evaluate_population(population, image, original_size)
        
        parents = select_parents(population, fitness_scores)
        
        offspring = crossover(parents)
        
        population = mutate(offspring, max_dim=max_dim)
        
        best_idx = np.argmax(fitness_scores)
        best_individual = population[best_idx]
        best_cr = fitness_scores[best_idx]
        best_rd = calc_data_red(best_cr)
        
        print(f"Generation {gen+1}, Best CR: {best_cr}, Best Block Size: {best_individual}")
    
    return best_individual, best_cr, best_rd

if __name__ == "__main__":
    image_path = 'test.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    binary_image = binary_image // 255
    
    best_individual, best_cr, best_rd = optimize_block_size(binary_image)
    
    best_p, best_q = best_individual
    print(f"Best block size: {best_p} x {best_q}")
    print(f"Compression Ratio (CR): {best_cr}")
    print(f"Relative Data Redundancy (RD): {best_rd}")
