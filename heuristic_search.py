import abc
import random
import math
import copy
import config
import compiler_flags
import enums
import debug
import individual
import collections
import internal_exceptions
import itertools
import os
from Queue import Queue
from threading import Thread

class SearchStrategy:
    """Abstract class for a search strategy"""
    
    @abc.abstractmethod
    def run(self):
        pass
    
    @abc.abstractmethod
    def summarise(self):
        pass

    @abc.abstractmethod
    def logall(self):
        pass
    
class GA(SearchStrategy):
    """Search using a genetic algorithm"""

    def logall(self):
        return
    
    def set_child_flags(self, child, the_flags, the_flag_values):
        for idx, flag in enumerate(the_flags):
            if flag in compiler_flags.PPCG.optimisation_flags:
                child.ppcg_flags[flag] = the_flag_values[idx]
            elif flag in compiler_flags.CC.optimisation_flags:
                child.cc_flags[flag] = the_flag_values[idx]
            elif flag in compiler_flags.CXX.optimisation_flags:
                child.cxx_flags[flag] = the_flag_values[idx]
            elif flag in compiler_flags.NVCC.optimisation_flags:
                child.nvcc_flags[flag] = the_flag_values[idx]
            else:
                assert False, "Unknown flag %s" % flag
                
    def set_sizes_flag(self, child, dominant_parent, submissive_parent):
        # We handle the crossover of the --sizes flag in a special manner as the
        # values of this flag are not simple scalar values
        the_sizes_flag = compiler_flags.PPCG.flag_map[compiler_flags.PPCG.sizes]
        child.ppcg_flags[the_sizes_flag] = compiler_flags.SizesFlag.crossover(self, 
                                                                              dominant_parent.ppcg_flags[the_sizes_flag], 
                                                                              submissive_parent.ppcg_flags[the_sizes_flag])
        
    
    def one_point(self, mother, father, children):
        """Implementation of 1-point crossover"""
        father_flags = father.all_flag_values()
        mother_flags = mother.all_flag_values()
        assert len(father_flags) == len(mother_flags)
        
        # Compute the crossover indices            
        point1 = 0
        point2 = random.randint(point1, len(mother_flags))
        point3 = len(mother_flags)
        
        child1_flags = []
        child1_flags.extend(mother_flags[point1:point2])
        child1_flags.extend(father_flags[point2:point3])
        child1 = individual.Individual()
        self.set_child_flags(child1, mother.all_flags(), child1_flags)
        self.set_sizes_flag(child1, mother, father)
        
        if children == 1:
            return [child1]
        
        child2_flags = []
        child2_flags.extend(father_flags[point1:point2])
        child2_flags.extend(mother_flags[point2:point3])
        child2 = individual.Individual()
        self.set_child_flags(child2, mother.all_flags(), child2_flags)
        self.set_sizes_flag(child2, father, mother)
        
        return [child1, child2]
          
    def two_point(self, mother, father, children):
        """Implementation of 2-point crossover"""
        father_flags = father.all_flag_values()
        mother_flags = mother.all_flag_values()
        assert len(father_flags) == len(mother_flags)
        
        # Compute the crossover indices            
        point1 = 0
        point2 = random.randint(point1, len(mother_flags))
        point3 = random.randint(point2, len(mother_flags))
        point4 = len(mother_flags)
        
        child1_flags = []
        child1_flags.extend(mother_flags[point1:point2])
        child1_flags.extend(father_flags[point2:point3])
        child1_flags.extend(mother_flags[point3:point4])
        child1 = individual.Individual()
        self.set_child_flags(child1, mother.all_flags(), child1_flags)
        self.set_sizes_flag(child1, mother, father)
        
        if children == 1:
            return [child1]
        
        child2_flags = []
        child2_flags.extend(father_flags[point1:point2])
        child2_flags.extend(mother_flags[point2:point3])
        child2_flags.extend(father_flags[point3:point4])
        child2 = individual.Individual()
        self.set_child_flags(child2, mother.all_flags(), child2_flags)
        self.set_sizes_flag(child2, father, mother)
        
        return [child1, child2]
    
    def select_parent(self, cumulative_fitnesses):
        # This implements roulette wheel selection
        for tup in cumulative_fitnesses:
            if tup[0] > random.uniform(0.0,1.0):
                return tup[1]
    
    def do_mutation(self, child):
        debug.verbose_message("Mutating child %d" % child.ID, __name__)
        for flag in child.ppcg_flags.keys():   
            if bool(random.getrandbits(1)):
                child.ppcg_flags[flag] = flag.random_value()
        for flag in child.cc_flags.keys():    
            if bool(random.getrandbits(1)):
                child.cc_flags[flag] = flag.random_value()
        for flag in child.cxx_flags.keys():    
            if bool(random.getrandbits(1)):
                child.cxx_flags[flag] = flag.random_value()
        for flag in child.nvcc_flags.keys():    
            if bool(random.getrandbits(1)):
                child.nvcc_flags[flag] = flag.random_value()
    
    def create_initial(self):
        new_population = []
        for i in range(0, config.Arguments.population):
            solution = individual.create_random()
            new_population.append(solution)
        return new_population
    
    def normalise_fitnesses(self, old_population):
        total_fitness = 0.0
        for individual in old_population:
            total_fitness += individual.fitness
        for individual in old_population:
            individual.fitness /= total_fitness
        old_population.sort(key=lambda x: x.fitness, reverse=True)
    
    def do_evolution(self, old_population):     
        # Normalise the fitness of each individual
        self.normalise_fitnesses(old_population)
        
        # Calculate a prefix sum of the fitnesses
        cumulative_fitnesses = []
        for idx, ind in enumerate(old_population):
            if idx == 0:
                cumulative_fitnesses.insert(idx, (ind.fitness, ind))
            else:
                cumulative_fitnesses.insert(idx, (ind.fitness + cumulative_fitnesses[idx-1][0], ind))
        
        # The new population     
        new_population = []
        
        if config.Arguments.random_individual:
            # Add a random individual as required
            solution = individual.create_random()
            new_population.append(solution)
        
        if config.Arguments.elite_individual:
            # Add the elite candidate as required
            try:
                fittest  = individual.get_fittest(old_population)
                clone    = copy.deepcopy(fittest)
                clone.ID = individual.Individual.get_ID()
                new_population.append(clone)
            except internal_exceptions.NoFittestException:
                pass
        
        # Add children using crossover and mutation
        while len(new_population) < len(old_population):
            crossover = getattr(self, config.Arguments.crossover)
            mother    = self.select_parent(cumulative_fitnesses)
            father    = self.select_parent(cumulative_fitnesses)
            # Create as many children as needed
            if len(new_population) < len(old_population) - 2:
                if random.uniform(0.0, 1.0) < config.Arguments.crossover_rate:
                    childList = crossover(mother, father, 2)
                    self.total_crossovers += 1
                else:
                    childList = [mother, father]
            else:
                if random.uniform(0.0, 1.0) < config.Arguments.crossover_rate:
                    childList = crossover(mother, father, 1)
                    self.total_crossovers += 1
                else:
                    if bool(random.getrandbits(1)):
                        childList = [mother]
                    else:
                        childList = [father]
            # Mutate
            for child in childList:
                if random.uniform(0.0, 1.0) < config.Arguments.mutation_rate:
                    self.total_mutations += 1
                    self.do_mutation(child)    
            # Add the children to the new population
            new_population.extend(childList)
        
        assert len(new_population) == len(old_population)
        return new_population    
    
    def run(self):        
        self.generations      = collections.OrderedDict()  
        self.total_mutations  = 0
        self.total_crossovers = 0
        
        state_random_population = "random_population"
        state_basic_evolution   = "basic_evolution"
        state_sizes_evolution   = "sizes_evolution"        
        current_state           = state_random_population
        legal_transitions       = set()
        legal_transitions.add((state_random_population, state_basic_evolution))
        legal_transitions.add((state_basic_evolution, state_sizes_evolution))
        legal_transitions.add((state_basic_evolution, state_basic_evolution))
        legal_transitions.add((state_sizes_evolution, state_basic_evolution))
        
        for generation in xrange(1, config.Arguments.generations+1):
            debug.verbose_message("%s Creating generation %d %s" % ('+' * 10, generation, '+' * 10), __name__)
            if current_state == state_random_population:
                self.generations[generation] = self.create_initial()
                next_state = state_basic_evolution
            elif current_state == state_basic_evolution:
                old_population = self.generations[generation-1]
                self.generations[generation] = self.do_evolution(old_population)
                next_state = state_basic_evolution
            elif current_state == state_sizes_evolution:
                debug.verbose_message("Now tuning individual kernel sizes", __name__)
                the_sizes_flag = compiler_flags.PPCG.flag_map[compiler_flags.PPCG.sizes]
                old_population = self.generations[generation-1]
                for individual in old_population:
                    individual.ppcg_flags[the_sizes_flag] = individual.size_data
                self.generations[generation] = self.do_evolution(old_population)
                legal_transitions.remove((state_basic_evolution, state_sizes_evolution))
                next_state = state_basic_evolution
            else:
                assert False, "Unknown state reached"
            
            # Generation created, now calculate the fitness of each individual
            for solution in self.generations[generation]:
                solution.run()
                
            if current_state == state_basic_evolution:
                # Decide whether to start tuning on individual kernel sizes in the next state
                if not config.Arguments.no_tune_kernel_sizes \
                and (state_basic_evolution, state_sizes_evolution) in legal_transitions \
                and bool(random.getrandbits(1)):
                    next_state = state_sizes_evolution
                    
            current_state = next_state
                    
    def summarise(self):
        print("%s Summary of %s %s" % ('*' * 30, __name__, '*' * 30))
        print("Total number of mutations:  %d" % (self.total_mutations))
        print("Total number of crossovers: %d" % (self.total_crossovers))
        print
        print("Per-generation summary")
        for generation, population in self.generations.iteritems():
            try:
                fittest = individual.get_fittest(population)
                debug.summary_message("The fittest individual from generation %d had execution time %f seconds" % (generation, fittest.execution_time)) 
                debug.summary_message("To replicate, pass the following to PPCG:")
                debug.summary_message(fittest.ppcg_cmd_line_flags, False)
            except internal_exceptions.NoFittestException:
                pass            

class Random(SearchStrategy):
    """Search using random sampling"""
    
    def run(self):
        self.individuals = []
        for i in xrange(1, config.Arguments.population+1):
            solution = individual.create_random()
            solution.run()
            self.individuals.append(solution)
    
    def summarise(self):
        print("%s Summary of %s %s" % ('*' * 30, __name__, '*' * 30))
        try:
            fittest = individual.get_fittest(self.individuals)
            debug.summary_message("The fittest individual had execution time %f seconds" % (fittest.execution_time)) 
            debug.summary_message("To replicate, pass the following to PPCG:")
            debug.summary_message(fittest.ppcg_cmd_line_flags, False)
        except internal_exceptions.NoFittestException:
            pass

    def logall(self):
        for i in self.individuals:
            debug.summary_message(i.ppcg_cmd_line_flags, False)
        return

compile_queue = Queue(10)
run_queue = Queue(10)

class CompileThread(Thread):
    def run(self):
        global compile_queue
        global run_queue
        while True:
            testcase = compile_queue.get()
            if isinstance(testcase, individual.EndOfQueue):
                run_queue.put(testcase)
                break

            testcase.ppcg()
            testcase.build()
            run_queue.put(testcase)

class RunThread(Thread):
    def __init__(self, num_threads):
        super(RunThread, self).__init__()
        self.num_threads = num_threads
        self.individuals = []

    def run(self):
        global run_queue
        best_time = float("inf")
        f = open(config.Arguments.results_file + ".log", 'a')
        f_iter = open('.lastiter', 'w')
        while True:
            #print('***run thread waiting')
            testcase = run_queue.get()
            if isinstance(testcase, individual.EndOfQueue):
                self.num_threads = self.num_threads - 1
                print('***remaining threads: ' + str(self.num_threads))
                if self.num_threads<=0:
                    try:
                       os.remove('.lastiter')
                       self.summarise()
                       self.logall()
                    except:
                       pass
                    print('***run thread exiting')
                    break
                continue
            #print('***run thread got job')
            testcase.binary(best_time)
            f_iter.seek(0)
            f_iter.write(str(testcase.get_ID()))

            if testcase.execution_time < best_time and testcase.execution_time != 0 and testcase.status == enums.Status.passed: 
                self.individuals.append(testcase)
                best_time = testcase.execution_time
                f.write("\n Best iter so far = \n")
                f.write(str(testcase))
                f.flush()

    def summarise(self):
        print("%s Summary of %s %s" % ('*' * 30, __name__, '*' * 30))
        try:
            fittest = individual.get_fittest(self.individuals)
            debug.summary_message("The fittest individual had execution time %f seconds" % (fittest.execution_time)) 
            debug.summary_message("To replicate, pass the following to PPCG:")
            debug.summary_message(fittest.ppcg_cmd_line_flags, False)
        except internal_exceptions.NoFittestException:
           pass

    def logall(self):
        print("%s Log of all runs %s" %('*' * 30, '*' * 30))
        for i in self.individuals:
            print(i)
            debug.summary_message(i.ppcg_cmd_line_flags, False)
        pass

class Exhaustive(SearchStrategy):
    """Exhaustive search all the values in the specified range or """
    """all combinations provided in explore-params.py file"""

    def readParamValues(self):
        f = open('explore-params.py', 'r')
        paramValues = eval(f.read())
        f.close()
        return paramValues

    def countConfigs(self, paramValues):
        n = 1
        for i in paramValues:
            n *= len(i)
        return n

    def createExhaConfigs(self):
        tile_size_lb = config.Arguments.tile_size_range[0] 
        tile_size_ub = config.Arguments.tile_size_range[1]
        if config.Arguments.only_powers_of_two:
            tile_size_range = [2**i for i in range(tile_size_lb, tile_size_ub)]
        else:
            tile_size_range = range(tile_size_lb, tile_size_ub)

        tile_sizes = itertools.product(tile_size_range, repeat=config.Arguments.tile_dimensions)
        
        block_size_lb = config.Arguments.block_size_range[0] 
        block_size_ub = config.Arguments.block_size_range[1] 
        if config.Arguments.only_powers_of_two:
            block_size_range = [2**i for i in range(block_size_lb, block_size_ub)]
        else:
            block_size_range = range(block_size_lb, block_size_ub)

        block_sizes = itertools.product(block_size_range, repeat=config.Arguments.block_dimensions)

        grid_size_lb = config.Arguments.grid_size_range[0] 
        grid_size_ub = config.Arguments.grid_size_range[1] 
        if config.Arguments.only_powers_of_two:
            grid_size_range = [2**i for i in range(grid_size_lb, grid_size_ub)]
        else:
            grid_size_range = range(grid_size_lb, grid_size_ub)

        grid_sizes = itertools.product(grid_size_range, repeat=config.Arguments.grid_dimensions)

        if config.Arguments.no_shared_memory:
            shared_mem = [True, False]
        else:
            shared_mem = [False]

        if config.Arguments.no_private_memory:
            private_mem = [True, False]
        else:
            private_mem = [False]

        paramValues = [tile_sizes, block_sizes, grid_sizes, shared_mem, private_mem]
        return paramValues

    def get_last_iter(self):
        if os.path.isfile(".lastiter"):
            print("found last iter")
            try:
                f_iter = open(".lastiter", 'r+')
                start_iter = int(f_iter.readline())
            except:
                start_iter = 0
                pass
            print("starting from test case = ", start_iter)
        else:
            start_iter = 0

        return start_iter

    def pipelineExec(self, combs):

        start_iter = self.get_last_iter()
        num_threads = config.Arguments.num_compile_threads
        for i in range(num_threads):
            t = CompileThread()
            t.daemon = True
            t.start()

        RunThread(num_threads).start()

        cnt = 0
        for conf in combs:
            if cnt < start_iter:
                cnt += 1
                continue
            print '---- Configuration ' + str(cnt) + ': ' + str(conf)
            cur = individual.create_test_case(conf[0], conf[1], conf[2], conf[3], conf[4])
            cur.set_ID(cnt)
            cnt += 1
            compile_queue.put(cur)

        for i in range(num_threads):
            compile_queue.put(individual.EndOfQueue()) # So every CompileThread fetches one EndOfQueue element
       
    def tile_size_multiple_filter(self, conf):
        tile_size = conf[0]
        block_size = conf[1]

        work_group_size = reduce(lambda x,y: x*y, block_size)
        if work_group_size > config.Arguments.max_work_group_size:
            return False

        if work_group_size < config.Arguments.min_work_group_size:
            return False

        mul_factor = 1
        for t, b in zip(tile_size, block_size):
            if t < b:
                return False
            if t % b != 0:
                return False
            mul_factor *= t/b

        if mul_factor > 36:
            return False

        return True

    def run(self):
        self.individuals = []

        if config.Arguments.params_from_file:
            paramValues = self.readParamValues()
        else:
            paramValues = self.createExhaConfigs()

        cnt = 0
        combs = itertools.product(*paramValues)


        if config.Arguments.filter_testcases:
            #Filter out only test cases based on heusristics such as tile size is multiple of block size etc.. 
            combs = filter(self.tile_size_multiple_filter, combs)
            #Filter out only test cases where shared memory is true
            combs = filter(lambda conf: conf[3] == True, combs)
            #Filter out only test cases where private memory is true
            combs = filter(lambda conf: conf[4] == True, combs)

        if config.Arguments.parallelize_compilation:
            self.pipelineExec(combs)
            return

        f = open(config.Arguments.results_file + ".log", 'a')
        start_iter = 0

        best_time = float("inf")
        #print 'Parameter values to be explored: ' + str(paramValues)
        #print 'Number of configurations: ' + str(self.countConfigs(paramValues))
        for conf in combs:
            if cnt < start_iter:
                cnt += 1
                continue
            print '---- Configuration ' + str(cnt) + ': ' + str(conf)
            cur = individual.create_test_case(conf[0], conf[1], conf[2], conf[3], conf[4]])
            cur.set_ID(cnt)
            cnt += 1
            cur.run(best_time)
            if cur.status == enums.Status.ppcgtimeout :
                f.write("\nppcg timeout")
                f.write(str(best_run))
                f.flush()
                continue
                
            if cur.execution_time == 0:
                continue

            if cur.execution_time < best_time and cur.status == enums.Status.passed:
                self.individuals.append(cur)
                best_time = cur.execution_time
                best_run = cur
                f.write("\n Best iter so far = "+ str(cnt) + "\n")
                f.write(str(best_run))
                f.flush()

    def summarise(self):
        print("%s Summary of %s %s" % ('*' * 30, __name__, '*' * 30))
        try:
            fittest = individual.get_fittest(self.individuals)
            debug.summary_message("The fittest individual had execution time %f seconds" % (fittest.execution_time)) 
            debug.summary_message("To replicate, pass the following to PPCG:")
            debug.summary_message(fittest.ppcg_cmd_line_flags, False)
        except internal_exceptions.NoFittestException:
           pass

    def logall(self):
        print("%s Log of all runs %s" %('*' * 30, '*' * 30))
        for i in self.individuals:
            print(i)
            debug.summary_message(i.ppcg_cmd_line_flags, False)
        pass

class SimulatedAnnealing(SearchStrategy):
   """Search using simulated annealing"""

   def acceptance_probability(self, currentEnergy, newEnergy, temperature):
        if newEnergy < currentEnergy:
            return 1.0
        return math.exp((currentEnergy - newEnergy) / temperature) 

   def logall(self):
        return

   def mutate_backend_flags(self, clone_flags, solution_flags):
        for the_flag in solution_flags.keys():   
            if bool(random.getrandbits(1)):
                idx    = the_flag.possible_values.index(solution_flags[the_flag])
                newIdx = (idx + 1) % len(the_flag.possible_values)
                clone_flags[the_flag] = the_flag.possible_values[newIdx]
    
   def mutate(self, solution):
        clone    = copy.deepcopy(solution)
        clone.ID = individual.Individual.get_ID()
        for the_flag in solution.ppcg_flags.keys():   
            if bool(random.getrandbits(1)):
                if isinstance(the_flag, compiler_flags.EnumerationFlag):
                    idx    = the_flag.possible_values.index(solution.ppcg_flags[the_flag])
                    newIdx = (idx + 1) % len(the_flag.possible_values)
                    clone.ppcg_flags[the_flag] = the_flag.possible_values[newIdx]
                else:
                    assert isinstance(the_flag, compiler_flags.SizesFlag)
                    clone.ppcg_flags[the_flag] = the_flag.permute(solution.ppcg_flags[the_flag])
                    
        self.mutate_backend_flags(clone.cc_flags, solution.cc_flags)
        self.mutate_backend_flags(clone.cxx_flags, solution.cxx_flags)
        self.mutate_backend_flags(clone.nvcc_flags, solution.nvcc_flags)
        return clone
    
   def run(self):        
        debug.verbose_message("Creating initial solution", __name__)
        current = individual.create_random()
        current.run()   
        self.fittest = current
        
        temperature = config.Arguments.initial_temperature
        for i in range(1, config.Arguments.cooling_steps+1):
            debug.verbose_message("Cooling step %d" % i, __name__)
            temperature *= config.Arguments.cooling
            for j in range(1, config.Arguments.temperature_steps+1):
                debug.verbose_message("Temperature step %d" % j, __name__)
                new = self.mutate(current)
                new.run()       
                if new.status == enums.Status.passed:     
                    if self.acceptance_probability(current.execution_time, new.execution_time, temperature):
                        current = new
                    if current.execution_time < self.fittest.execution_time:
                        self.fittest = current
    
   def summarise(self):
        debug.summary_message("The final individual had execution time %f seconds" % (self.fittest.execution_time)) 
        debug.summary_message("To replicate, pass the following to PPCG:")
        debug.summary_message(self.fittest.ppcg_cmd_line_flags, False)
        
