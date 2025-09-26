#TABU SEARCH WITH WEIGHTED ONE MAX PROBLEM
import matplotlib.pyplot
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import csv
import re
import os
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#DEFINIZIONE ONE MAX PROBLEM
#max_iterations è il numero massimo di iterazioni della TS
# x è la stringa binaria di dimensione l 
# w è il vettore di pesi associato a x 

max_iterations=100
x=[]
w=[]
l=0
h=0

try:
    l = int(input("Inserire un numero l per definire la lunghezza della stringa binaria:"))
    h = int(input("Inserire un numero h per definire il range [-h,h] entro cui assegnare i pesi: "))
except:
    print(TypeError, "Inserire un numero intero!")

#---------------------------------------------------------------------------------------------------------------------
#Fitness function
def fitness(x,w):
    f=0
    for i in range(len(x)):
        f=f+x[i]*w[i]
    return f
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#INIZIALIZZAZIONE PESI


def init_weights():
    for _ in range(l):
        w.append(round(random.uniform(-h,h),2))

def read_weights_from_file(f:str):

    weights = []
    try:
        with open(filename, mode='r', newline='') as file:
            # Specifica il delimitatore come spazio
            reader = csv.reader(file, delimiter=' ')
            # Skippa la prima riga "Weights:"
            next(reader)
            for row in reader:
                for value in row:
                    stripped_value = (value.strip())
                    if stripped_value: 
                        weights.append(float(stripped_value))
    except FileNotFoundError:
        print(f"Errore: il file {filename} non esiste.")
    except Exception as e:
        print(f"Errore nella lettura del file: {e}")

    file.close()
    return weights

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#INIZIALIZZAZIONE DELLA SOLUZIONE INIZIALE



#inizializzazione della soluzione iniziale pari a una stringa di soli zeri
def initial_solution():
    return [0] * l


#inizializzazione della soluzione iniziale pari a una stringa di 0 e 1 random
def initial_random_solution():
    return [random.randint(0, 1) for _ in range(l)]


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#TABU SEARCH IMPLEMENTATION WITH WEIGHTED ONE MAX PROBLEM
#bisogna trovare la stringa di 1 nella stringa binaria che permette di massimizzare la fitness function

#l'idea è di flippare bit in base a una probabilità associata al peso

def get_neighbors_single_flip(solution,w):
    neighbors = []
    for i in range(len(solution)):
        neighbor = solution.copy()  # Crea una copia della soluzione corrente
        neighbor[i] = 1 - neighbor[i]  # Inverti il bit i
        neighbors.append(neighbor)
    return neighbors

def get_neighbors_multiple_flip(solution, w):
    num_flips=2

    neighbors = []
    for _ in range(10):  # Genera 10 vicini casuali
        neighbor = solution.copy()
        flip_indices = random.sample(range(len(solution)), num_flips)
        for i in flip_indices:
            neighbor[i] = 1 - neighbor[i]  # Inverti i bit selezionati
        neighbors.append(neighbor)
    return neighbors

def get_neighbors_probability(solution,w):
    neighbors = []

    for i in range(len(solution)):
        neighbor = solution.copy()

        # Calcola una probabilità di flip basata sul valore del peso
        weight = w[i]

        if weight > 0:
            # Pesi positivi: vogliamo incentivare che i bit siano 1
            if solution[i] == 0:
                flip_probability = (h - weight) / h  # Maggiore il peso, minore la probabilità di lasciare 0
                #flip_probability = 1
            else:
                flip_probability = weight / h  # Maggiore il peso, minore la probabilità di flippare 1 a 0
                #flip_probability= 0
        
        
        elif weight < 0:
            # Pesi negativi: vogliamo incentivare che i bit siano 0
            if solution[i] == 0:
                flip_probability = (-weight) / h  # Minore il peso (più negativo), minore la probabilità di flippare 0 a 1
                #flip_probability=0
            else:
                flip_probability = (h + weight) / h  # Pesi negativi, minore la probabilità di lasciare 1
                #flip_probability = 1
        
        # Flip del bit con una certa probabilità
        if random.random() < flip_probability:
            neighbor[i] = 1 - neighbor[i]  # Inverti il bit i

        neighbors.append(neighbor)
    #print("All neighbors:", neighbors)
    return neighbors


def objective_function(neighbor):
    f=0
    for i in range(len(neighbor)):
        f=f+neighbor[i]*w[i]
    return f


def aspiration_criteria(neighbor_fitness, best_fitness):
    return neighbor_fitness > best_fitness

def get_neighbors(get_neighbors_strategy, current_solution,w):
    if (get_neighbors_strategy=='single'):
        return get_neighbors_single_flip(current_solution,w)
    elif(get_neighbors_strategy=='multiple'):
        return get_neighbors_multiple_flip(current_solution,w)
    elif(get_neighbors_strategy=='probability'):
        return get_neighbors_probability(current_solution, w)



def tabu_search(initial_solution, max_iterations, tabu_list_min_size, tabu_list_max_size, get_neighbors_strategy='probability'):
    best_solution = initial_solution
    current_solution = initial_solution
    tabu_list = []
    tabu_list_actual_size=(tabu_list_max_size-tabu_list_min_size)/2
    no_improvement_counter=0
    max_no_improvement_counter=5
    tabu_increment=2
    threshold=5

    #x è la soluzione iniziale [0,0,0,...,0]
    print("initial solution: ", best_solution)

    for _ in range(max_iterations):
        neighbors = get_neighbors(get_neighbors_strategy, current_solution,w)

        best_neighbor = None
        best_neighbor_fitness = float('-inf')

        for neighbor in neighbors:
            if neighbor not in tabu_list or aspiration_criteria(neighbor_fitness, best_neighbor_fitness):
                neighbor_fitness = objective_function(neighbor)
                if (neighbor_fitness > best_neighbor_fitness):
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    print("Best fitness value: ",best_neighbor_fitness)

        if (best_neighbor is None):
            print("No best neighbor found")
            break

        current_solution=best_neighbor
        tabu_list.append(best_neighbor)

        if (len(tabu_list)>tabu_list_actual_size):
            tabu_list.pop(0)
        
        best_neighbor_fitness=objective_function(best_neighbor)
        best_solution_fitness=objective_function(best_solution)


        if (best_neighbor_fitness>best_solution_fitness):
            print("updating the best solution...")
            best_solution=best_neighbor
        else:
            no_improvement_counter=+1

        if (no_improvement_counter>=max_no_improvement_counter):
            tabu_list_actual_size=min(tabu_list_actual_size+tabu_increment,tabu_list_max_size)
            print("Nessun miglioramento trovato, incremento della tabu list size ...")
        elif(best_neighbor_fitness>best_solution_fitness-threshold):
            tabu_list_actual_size=max(tabu_list_actual_size-tabu_increment, tabu_list_min_size)
            print("Miglioramenti frequenti trovati, riduzione della tabu list size ...")

    return best_solution

init_weights()
start=time.time()
best=tabu_search(initial_solution(),max_iterations,3,18,'multiple')
end=time.time()

print("Tempo necessario per trovare la soluzione:",round(end-start,5),"secondi")
print("Pesi attuali:",w)
print("Migliore soluzione trovata:", best)
print("Fitness della migliore soluzione:", objective_function(best))


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#FASE DI EVALUATION

input("Premere invio per iniziare fase di evaluation")
##EVALUATION
plt.figure(figsize=(8, 6))
str_len=[2,10,100,200,500,1000]
w_range=[2,10,100,200,500,1000]
w=[]
x=[]

input("Evaluation della strategia single flip con l=[2,10,100,200,500,1000] e range h=[2,10,100,200,500,1000]")
time_elapsed_single=[]

for i in range(len(str_len)):
    l=str_len[i]
    h=w_range[i]
    init_weights()
    start=time.time()
    best=tabu_search(initial_solution(),max_iterations,3,18,'single')
    end=time.time()
    time_elapsed_single.append(end-start)
    print("Con l=",l,"si ha soluzione",best, "in tempo", end-start)

plt.title('Performance single bit flip strategy ')
plt.xlabel('Lunghezza della stringa binaria')
plt.ylabel('Tempo impiegato per la soluzione')
plt.plot(np.array(str_len), np.array(time_elapsed_single),color='green', linestyle='--', marker='o')
plt.show()

input("Evaluation della strategia multiple flip con l=[2,10,100,200,500,1000] e range h=[2,10,100,200,500,1000]")
time_elapsed_multiple=[]
for i in range(len(str_len)):
    l=str_len[i]
    h=w_range[i]
    init_weights()
    start=time.time()
    best=tabu_search(initial_solution(),max_iterations,3,18,'multiple')
    end=time.time()
    time_elapsed_multiple.append(end-start)
    print("Con l=",l,"si ha soluzione in tempo", end-start)
    input()

plt.title('Performance multiple bit flip strategy ')
plt.xlabel('Lunghezza della stringa binaria')
plt.ylabel('Tempo impiegato per la soluzione')
plt.plot(np.array(str_len), np.array(time_elapsed_multiple),color='green', linestyle='--', marker='o')
plt.show()

input("Evaluation della strategia con probabilità associate ai pesi con l=[2,10,100,200,500,1000] e range h=[2,10,100,200,500,1000]")
time_elapsed_probability=[]
for i in range(len(str_len)):
    l=str_len[i]
    h=w_range[i]
    init_weights()
    start=time.time()
    best=tabu_search(initial_solution(),max_iterations,3,18,'probability')
    end=time.time()
    time_elapsed_probability.append(end-start)
    print("Con l=",l,"si ha soluzione",best, "in tempo", end-start,"secondi")
    input()

plt.title('Performance weights-based probability flip strategy')
plt.xlabel('Lunghezza della stringa binaria')
plt.ylabel('Tempo impiegato per la soluzione')
plt.plot(np.array(str_len), np.array(time_elapsed_probability),color='green', linestyle='--', marker='o')
plt.show()

input("Grafico di comparison")

plt.plot(np.array(str_len), np.array(time_elapsed_single),color='green', linestyle='--', marker='o', label='Single flip')
plt.plot(np.array(str_len), np.array(time_elapsed_multiple),color='red', linestyle='--', marker='o',label='Multiple flip')
plt.plot(np.array(str_len), np.array(time_elapsed_probability),color='blue', linestyle='--', marker='o',label='Weight-based probability flip')

plt.title('Performance comparison')
plt.xlabel('Lunghezza della stringa binaria')
plt.ylabel('Tempo impiegato per la soluzione')
plt.grid(True)

plt.legend()
plt.show()


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#TEST TABU SEARCH SU ISTANZE DA FILE TXT ESTERNO


def extract_l_h_from_filename(filename):
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

def take_inputs_from_folder():
    files = os.listdir('.')
    txt_files = [f for f in files if f.startswith("Instance") and f.endswith('.txt')]
    return txt_files


os.system('cls')
print("FASE DI EVALUATION DELLE ISTANZE DEL PROBLEMA IN FORMATO .TXT")

files=take_inputs_from_folder()
print("-------------------------------------------")
print("FILES:")
for i in range(len(files)):
    print(files[i])
print("-------------------------------------------")
input()


for i in range(len(files)):
    filename=files[i]
    l,_,h = extract_l_h_from_filename(filename)
    w=[]
    x=[]
    print("FILENAME:", filename)
    input()
    w = read_weights_from_file(filename)
    print("SINGLE FLIP STRATEGY")
    input()
    start=time.time()
    best=tabu_search(initial_solution(),max_iterations,5,25,'single')
    end=time.time()
    opt_sol=objective_function(best)
    input()
    print("Pesi attuali:",w,"\n")
    print("Migliore soluzione trovata:", best,"\n")
    print("Fitness della migliore soluzione:", objective_function(best),"\n")
    print("Tempo necessario per trovare la soluzione:",round(end-start,5),"secondi\n")
    input()
    
    sol=[]
    err=[]
    sum=0

    for i in range(10):
        print("MULTIPLE FLIP STRATEGY")
        input()
        start=time.time()
        best=tabu_search(initial_solution(),max_iterations,5,25,'multiple')
        end=time.time()
        input()
        print("Pesi attuali:",w,"\n")
        print("Migliore soluzione trovata:", best,"\n")
        print("Fitness della migliore soluzione:", objective_function(best),"\n")
        print("Tempo necessario per trovare la soluzione:",round(end-start,5),"secondi\n")
        input()
        sol.append(objective_function(best))
        err.append((sol[i]- opt_sol)/opt_sol)
        sum=sum+err[i]
        print("Errore relativo rispetto a soluzione ottima:",err[i])
        input()

    
    print("Errore relativo medio su ",i,"iterazioni:",sum/i)
    sol=[]
    err=[]
    sum=0
    input()
    for i in range(10):
        print("WEIGHT-BASED PROBABILITY FLIP STRATEGY")
        input()
        start=time.time()
        best=tabu_search(initial_solution(),max_iterations,5,25,'probability')
        end=time.time()
        print("Pesi attuali:",w,"\n")
        print("Migliore soluzione trovata:", best,"\n")
        print("Fitness della migliore soluzione:", objective_function(best),"\n")
        print("Tempo necessario per trovare la soluzione:",round(end-start,5),"secondi\n")
        input()
        sol.append(objective_function(best))
        err.append((sol[i]- opt_sol)/opt_sol)
        sum=sum+err[i]
        print("Errore relativo rispetto a soluzione ottima:",err[i])
        input()
    print("Errore relativo medio su ",i,"iterazioni:",sum/i)
    input()
    sum=0
    err=[]
    sol=[]