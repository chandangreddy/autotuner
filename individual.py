import timeit
import os
import re
import debug
import compiler_flags
import config
import enums
import collections
import subprocess
import threading
import internal_exceptions
import time

class EndOfQueue:
    def __init__(self):
        pass



def get_fittest(population):
    fittest = None
    for individual in population:
        if individual.status == enums.Status.passed:
            if fittest:
                if individual.fitness > fittest.fitness:
                    fittest = individual
            else:
                fittest = individual
    if not fittest:
        raise internal_exceptions.NoFittestException("None of the individuals among this population completed successfully, hence there is no fittest individual")
    return fittest

def create_test_case(tile_size, block_size, grid_size, shared_mem=True, private_mem=True, fusion='max'):
    individual = Individual()   
    per_kernel_size_info = collections.OrderedDict()
    per_kernel_size_info[compiler_flags.SizesFlag.ALL_KERNELS_SENTINEL] = compiler_flags.SizeTuple(tile_size, block_size, grid_size)

    #for flag in compiler_flags.PPCG.optimisation_flags:
    #    print(flag)

    #TODO: Get a better way of getting size_data_flag
    flag = compiler_flags.PPCG.optimisation_flags[4]
    individual.ppcg_flags[flag] = per_kernel_size_info 

    if not shared_mem:
        flag = compiler_flags.PPCG.optimisation_flags[0]
        #individual.ppcg_flags[flag] = compiler_flags.EnumerationFlag(flag) 
        individual.ppcg_flags[flag] = True 


    if not private_mem:
        flag = compiler_flags.PPCG.optimisation_flags[7]
        #individual.ppcg_flags[flag] = compiler_flags.EnumerationFlag(flag) 
        individual.ppcg_flags[flag] = True 

    #Set isl fusion flag
    flag = compiler_flags.PPCG.optimisation_flags[6]
    individual.ppcg_flags[flag] = fusion
    #string  = individual.ppcg_flags[flag].get_command_line_string(1024)
    #print(string)
    #print("end")
    return individual


def create_random():
    individual = Individual()   
    for flag in compiler_flags.PPCG.optimisation_flags:
        print(flag)
        individual.ppcg_flags[flag] = flag.random_value()
    for flag in compiler_flags.CC.optimisation_flags:
        individual.cc_flags[flag] = flag.random_value()
    for flag in compiler_flags.CXX.optimisation_flags:
        individual.cxx_flags[flag] = flag.random_value()
    for flag in compiler_flags.NVCC.optimisation_flags:
        individual.nvcc_flags[flag] = flag.random_value()
    return individual

class Individual:
    """An individual solution in a population"""
    
    ID = 0
    @staticmethod
    def get_ID_init():
        Individual.ID += 1
        return Individual.ID
    
    def file_name(self):
        return 'testcase'+str(self.ID)
        #return 'gemm'

    def set_ID(self, num):
        self.ID = num  

    def get_ID(self):
        return self.ID 
    
    def __init__(self):
        self.ID               = Individual.get_ID_init()
        self.ppcg_flags       = collections.OrderedDict()
        self.cc_flags         = collections.OrderedDict()
        self.cxx_flags        = collections.OrderedDict()
        self.nvcc_flags       = collections.OrderedDict()
        self.status           = enums.Status.failed
        self.execution_time   = float("inf") 
        self.num = 0
        
    def all_flags(self):
        return self.ppcg_flags.keys() + self.cc_flags.keys() + self.cxx_flags.keys() + self.nvcc_flags.keys()
    
    def all_flag_values(self):
        return self.ppcg_flags.values() + self.cc_flags.values() + self.cxx_flags.values() + self.nvcc_flags.values()
            
    def run(self, timeout):
        try:
            self.compile(timeout)
            if self.status == enums.Status.passed:
                # Fitness is inversely proportional to execution time
                if self.execution_time == 0:
                    self.fitness = float("inf")
                else:
                    self.fitness = 1/self.execution_time 
                debug.verbose_message("Individual %d: execution time = %f, fitness = %f" \
                                      % (self.ID, self.execution_time, self.fitness), __name__) 
            else:
                self.fitness = 0
        except internal_exceptions.FailedCompilationException as e:
            debug.exit_message(e)
            
            
    def checkforpause(self):
        while(1):
            if os.path.isfile('.pause'):
                print("Auto tuning paused, remove .pause to restart")
                time.sleep(20)
            else:
                #print("Auto tuning restarted")
                break

    def compile(self, timeout=2):
        self.checkforpause()
        sucess=self.ppcg_with_timeout()
        if not sucess:
            return
        self.build()
        self.run_with_timeout(timeout)

    def ppcg(self):
        self.ppcg_cmd_line_flags = "--target=%s --dump-sizes %s" % (config.Arguments.target, 
                                                                    ' '.join(flag.get_command_line_string(self.ppcg_flags[flag]) for flag in self.ppcg_flags.keys()))
        
        os.environ["AUTOTUNER_PPCG_FLAGS"] = self.ppcg_cmd_line_flags

        if config.Arguments.target == enums.Targets.cuda:
            cmd = config.Arguments.ppcg_cmd + ' '+self.ppcg_cmd_line_flags+' -o '+self.file_name()
        else:
            cmd = config.Arguments.ppcg_cmd + ' '+self.ppcg_cmd_line_flags+' -o '+self.file_name()+'_host.c'

        debug.verbose_message("Running '%s'" % cmd, __name__)
        #debug.verbose_message("Running '%s'" % self.ppcg_cmd_line_flags , __name__)
        start  = timeit.default_timer()
        self.ppcg_proc   = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE)  
        stderr = self.ppcg_proc.communicate()[1]
        end    = timeit.default_timer()
        config.time_PPCG += end - start
        if self.ppcg_proc.returncode:
            raise internal_exceptions.FailedCompilationException("FAILED: '%s'" % config.Arguments.ppcg_cmd)         
        # Store the sizes used by PPCG
        self.size_data = compiler_flags.SizesFlag.parse_PPCG_dump_sizes(stderr)
        

    def ppcg_with_timeout(self, timeout=200):
        thread = threading.Thread(target=self.ppcg)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            print("Timeout: terminating the ppcg ")
            self.ppcg_proc.terminate()
            thread.join(timeout)
            self.status = enums.Status.ppcgtimeout
            return False
        return True

    def build(self):
        if config.Arguments.target == enums.Targets.cuda:
            build_cmd = config.Arguments.build_cmd + ' ' + self.file_name()+ '_host.cu ' + self.file_name()+ '_kernel.cu '+ '-o '+ self.file_name()+'.exe'
        else:
            build_cmd = config.Arguments.build_cmd + ' ' + self.file_name()+ '_host.c ' + '-o '+ self.file_name()+'.exe'
        debug.verbose_message("Running '%s'" % build_cmd, __name__)
        start  = timeit.default_timer()
        proc   = subprocess.Popen(build_cmd, shell=True)  
        stderr = proc.communicate()[1]     
        end    = timeit.default_timer()
        config.time_backend += end - start
        if proc.returncode:
            raise internal_exceptions.FailedCompilationException("FAILED: '%s'" % config.Arguments.build_cmd)


    
    def deleteFile(self, fileName):
        try:
            if os.path.exists(fileName):
                os.remove(fileName)
        except:
            pass

    def binary(self):
        #time_regex = re.compile(r'^(\d*\.\d+|\d+)$')
        print config.Arguments.execution_time_regex
        time_regex = re.compile(config.Arguments.execution_time_regex)
        total_time = 0.0
        status     = enums.Status.passed
        for run in xrange(1,config.Arguments.runs+1):
            run_cmd = './'+self.file_name()+'.exe'
            #run_cmd = config.Arguments.run_cmd
            debug.verbose_message("Run #%d of '%s'" % (run, run_cmd), __name__)
            start = timeit.default_timer()
            self.proc  = subprocess.Popen(run_cmd, shell=True, stdout=subprocess.PIPE)    
            stdout, stderr = self.proc.communicate()
            end   = timeit.default_timer()
            if self.proc.returncode:
                sper_kernel_size_infotatus = enums.Status.failed
                debug.warning_message("FAILED: '%s'" % config.Arguments.run_cmd)
                continue
            if config.Arguments.execution_time_from_binary:
                if not stdout:
                    raise internal_exceptions.BinaryRunException("Expected the binary to dump its execution time. Found nothing")
                nmatchedlines = 0
                for line in stdout.split(os.linesep):
                    line    = line.strip()
                    matches = time_regex.findall(line)
                    if matches:
                        nmatchedlines += 1
                        try:
                            total_time += float(matches[0])
                        except:
                            raise internal_exceptions.BinaryRunException("Execution time '%s' is not in the required format" % matches[0])
                if nmatchedlines == 0:
                    raise internal_exceptions.BinaryRunException("Regular expression did not match anything on the program's output")
            else:
                total_time += end - start
        self.status = status
        config.time_binary += total_time
        self.execution_time = total_time/config.Arguments.runs

        self.deleteFile(self.file_name()+'.exe')
        self.deleteFile(self.file_name()+'_host.c')
        self.deleteFile(self.file_name()+'_host_kernel.cl')
        self.deleteFile(self.file_name()+'_host.cu')
        self.deleteFile(self.file_name()+'_kernel.cu')
        self.deleteFile(self.file_name()+'_kernel.hu')
        self.deleteFile(self.file_name()+'_host_kernel.hu')
        self.deleteFile(self.file_name()+'_host_kernel.h')
        self.deleteFile(self.file_name())

 
    def run_with_timeout(self, timeout=2):
        print "executing task " + str(self.ID)
        timeout = config.Arguments.timeout_ppcg
        try:
            thread = threading.Thread(target=self.binary)
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                print("Timeout: terminating the procs")
                self.proc.terminate()
                thread.join()
                self.status = enums.Status.timeout
        except:
            print("Exception running"+str(self.ID))
            self.status = enums.Status.timeout

        return
        
               
    def __str__(self):
        return "ID %4d: execution time = %3f, ppcg = %s, status = %s" % (self.ID, self.execution_time, self.ppcg_cmd_line_flags, self.status)
    
