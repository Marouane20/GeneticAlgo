#https://github.com/Marouane20/GeneticAlgo
import numpy as np

#** 2x2 + 4x − 4 = 0
#** x = −1 + √3  x = −1 − √3

symbol = np.array([2,4,-4])
print(symbol)
#Initialization
#** Init a matrix with random numbers between 0 and 30  
#** every ligne represent a chromosome
def init(num_lignes,num_chrom):
    #np.random.randint(0,30,(num_chrom,num_lignes))
    s = np.random.uniform(0,100,(num_chrom,num_lignes))
    #s[:,0] = s[:,0] **2 
    #s=np.insert(s,0,s[:,0]**2,axis=1)
    return s

#Evaluation
#**Compute the objective function value for each chromosome 

#a = np.array([[1,2],[3,4]])
#b = np.array([2,4,-4])
#print(np.insert(a,len(b),1,axis=1))

def objectif_function(matrix,symbols):

   # return np.abs(
   #    np.sum(
   #     np.multiply(
    #        np.insert(matrix,np.size(matrix,1),1,axis=1),symbols
     #       ),1
     #   )
    #)
    avg = np.average(matrix,axis=1)
    print("avg ",avg)
    arr = [symbols[0]*a**2+symbols[1]*a+symbols[2] for a in avg]
    return np.abs(
       np.array(arr)
    )   
     
matrix = init(5,6)
print(matrix)
f = objectif_function(matrix,symbol)
print("objectif function ",f)

#Selection 
#** To compute fitness probability we must compute the fitness of each chromosome.
#** To avoid divide by zero problem, the value of F_obj is added by 1.

#Fitness function Fitness[i] = 1 / (1+F_obj[i])
def fitness(obj_f):
    return np.divide(1,obj_f+1)

obj_f =fitness(f)
print(obj_f)

total = np.sum(obj_f)
print(total)

#Propability for each chromosome P[i] = Fitness[i] / Total
def probability(obj_f):
    return np.divide(obj_f,np.sum(obj_f))

prob = probability(obj_f)
print(prob)

#Selection using roulette wheel 
#** first calculate cumulative_propability
def cumulative_propability(prob):
    return np.cumsum(prob)

cumpro = cumulative_propability(prob)
print(cumpro)

#**Random numbers 
def random_number(prob):
    return np.random.random(np.size(prob))

rand = random_number(prob)
print(rand)

#** Selection 
def new_population(population,random_number,cumulative_propability):
    lis =[]
    print("lis ",lis)
    j=0
    for i in random_number:
        #print(i)
        while cumulative_propability[j]<= i  :
            j+=1
        lis.append(population[j,:])
        j=0

    return np.array(lis)

new_p = new_population(matrix,rand,cumpro) 
print("new_p",new_p)

#Crossover 
#** Parent chromosome which will mate is randomly selected 
#** and the number of mate Chromosomes is controlled using 
#**crossover_rate %(ρc) parameters.

def select_parents(crossover_rate,new_population):
    rand = np.random.random(np.size(new_population,axis = 0))
    print("rand  ",rand)
    return np.where(rand<crossover_rate)[0]

parents = select_parents(0.50,new_p)
print("parents ",parents)
print(np.size(parents))

#Crossover position 
#**generating random numbers between 1 to (length of Chromosome – 1).

def crossover(indices,new_population):
    comb =list()
    arr = list()
    print("new p ",new_population)
    for i in range(0,len(indices)-1) :
        j = i+1
        comb.append((indices[i],indices[j]))
        try:
            comb.append((indices[j+1],indices[i]))  
        except IndexError :
            break
    print("Combinations ",comb)
    [print( a) for a,b in comb]
    for a,b in comb:
        rand = np.random.randint(1,np.size(new_population,axis=1))
        print('len = ',rand)
        print("stack ",np.hstack((new_population[a,0:rand].copy(),new_population[b,rand:].copy())))
        arr.append(np.hstack((new_population[a,0:rand].copy(),new_population[b,rand:].copy())))
        
    print("arr " ,arr)
    arr = np.array(arr)
    print("arr " ,arr)
    i=0
    for a,b in comb :
        new_population[a] = arr[i]
        i+=1
    return new_population

cr = crossover(parents,new_p)
print("P after cross ",cr)

#Mutation
#**Mutation process is done by replacing the gen at random position with a new value.

#Generate random number between 0 and total number of Gen in population
def mutation(mutation_rate,new_population):
    total_number_of_gen = np.size(new_population)
    number_of_mutations = int(round(mutation_rate*total_number_of_gen))
    print(total_number_of_gen)
    print("number_of_mutations ",number_of_mutations)

    for i in range(0,number_of_mutations):
        rand = np.random.randint(0,total_number_of_gen)
        new_population.itemset(rand,np.random.uniform(0,15)) 
    return new_population

mu = mutation(0.2,cr)
print(mu)

def evaluation(new_population,symbols):

    return objectif_function(new_population,symbols)

ev = evaluation(mu,symbol)
print("Evaluation " ,ev)
print("Population ",mu)

def result(arr_symbols,num_lignes,num_chrom,num_iterations,pc,pm):

    matrix = init(num_lignes,num_chrom)
    print(matrix)
    for i in range(0,num_iterations):
        print("***** Itération N ",i," ******" )
        f = objectif_function(matrix,arr_symbols)
        
        obj_f =fitness(f)

        prob = probability(obj_f)

        cumpro = cumulative_propability(prob)

        rand = random_number(prob)

        new_p = new_population(matrix,rand,cumpro) 

        parents = select_parents(pc,new_p)
       
        cr = crossover(parents,new_p)
       
        mu = mutation(pm,cr)
       
        ev = evaluation(mu,arr_symbols)

        #print("Population ",mu)

        matrix = mu

    avg = np.average(matrix,axis=1)
    #print("avg ",avg)
    rnd = np.around(avg,decimals=4)
    # print("round ",rnd)
    best_solutions = np.array(np.unique(rnd,return_counts=True,return_index=True))
    print("best solutions ",best_solutions)
    print("Best Solution ",best_solutions[0,np.argwhere(best_solutions[2]==np.max(best_solutions[2]))][0,0])

    return [best_solutions,best_solutions[0,np.argwhere(best_solutions[2]==np.max(best_solutions[2]))][0,0]]


if __name__ == "__main__":
    
    #** 2x2 + 4x − 4 = 0
    #** x = −1 + √3  x = −1 − √3
    symbol = np.array([2,4,-4])
    #symbol = np.array([1,2,-3])
    print(symbol)
    rs = result(symbol,4,10,150,0.33,0.1)
    print(rs[0],rs[1])
    # matrix = init(4,10)
    # print(matrix)
    # for i in range(0,150):
    #     print("***** Itération N ",i," ******" )
    #    f = objectif_function(matrix,symbol)
        
    #     obj_f =fitness(f)

    #     prob = probability(obj_f)

    #     cumpro = cumulative_propability(prob)

    #     rand = random_number(prob)

    #     new_p = new_population(matrix,rand,cumpro) 

    #     parents = select_parents(0.33,new_p)
       
    #     cr = crossover(parents,new_p)
       
    #     mu = mutation(0.1,cr)
       
    #     ev = evaluation(mu,symbol)

    #     print("Population ",mu)

    #     matrix = mu

    # avg = np.average(matrix,axis=1)
    # print("avg ",avg)
    # rnd = np.around(avg,decimals=4)
    # print("round ",rnd)
    # best_solutions = np.array(np.unique(rnd,return_counts=True,return_index=True))
    # print("best solutions ",best_solutions)
    # print("Best Solution ",best_solutions[0,np.argwhere(best_solutions[2]==np.max(best_solutions[2]))][0,0])

print('hello world')

