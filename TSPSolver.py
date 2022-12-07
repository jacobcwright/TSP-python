#!/usr/bin/python3

from copy import deepcopy
from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
from itertools import count
tiebreaker = count()


class TSPSolver:
	def __init__(self, gui_view):
		self._scenario = None

	def setupWithScenario(self, scenario):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	# worst case O(n!) time and O(n) space
	def defaultRandomTour(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy(self, time_allowance=60.0):
		pass

	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy(self, time_allowance=60.0):
		bssf = None
		self.cities = self._scenario.getCities()
		self.numCities = len(self.cities)
		results = {}

		start = time.time()
		while time.time() - start < time_allowance:
			for i in range(self.numCities):
				city = self.cities[i]
				path = []
				visited = set()
				path.append(city)
				visited.add(city)

				while len(path) < self.numCities:
					# get the next city
					nextCity = self.getNextCity(city, filter(lambda c: c not in visited, self.cities))
					if city.costTo(nextCity) == np.inf:
						break
					path.append(nextCity)
					visited.add(nextCity)
					city = nextCity

				if len(path) != self.numCities:
					continue

				bssf = TSPSolution(path)
				end = time.time()
				results['cost'] = bssf.cost
				results['time'] = end - start
				results['count'] = 1
				results['soln'] = bssf
				results['max'] = None
				results['total'] = None
				results['pruned'] = None
				return results
		return results


	def getNextCity(self, currentCity, cityList):
		cityCosts = {}
		for city in cityList:
			cityCosts[city] = currentCity.costTo(city)
		
		return min(cityCosts, key=cityCosts.get)


	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
	# O(k*n^2) time and space where k is the number of states and n is the number of cities
	# This is ignoring the time it takes to get the initial bssf from random tour.
	def branchAndBound(self, time_allowance=60.0):
		# initialize variables (O(1))
		self.cities = self._scenario.getCities()
		self.numCities = len(self.cities)
		heap = []
		bestsFound = 0
		queueSize = 0
		totalStates = 0
		pruned = 0
		solutions = 0
		
		# get initial bssf using randomTour, potentially (O(n!))
		bssf = self.defaultRandomTour(time_allowance)['soln']
		solutions += 1
		self.lowestBound = bssf.cost

		reducedMatrix, cost = self.getBaseMatrix(self.cities)
		# Tuple is (cost, counter, matrix, currentCity, remainingCities, path) where counter is used to break ties
		heapq.heappush(heap, (cost, next(tiebreaker), reducedMatrix, self.cities[0], self.cities[1:], [self.cities[0]._index]))

		queueSize += 1
		totalStates += 1

		start = time.time()
		# runs until time runs out or the heap is empty (k)
		while len(heap) and time.time() - start < time_allowance:
			# get next cheapest city
			nextCity = heapq.heappop(heap)

			# if the cost is greater than the current best, prune
			if nextCity[0] >= self.lowestBound:
				pruned += 1
				totalStates += 1
				continue
			else:
				# if the cost is less than the current best, try to add the next city
				for city in nextCity[4]:
					if self._scenario._edge_exists[nextCity[3]._index][city._index]:
						# get potential new path -- O(n) time and O(n^2) space
						potential = self.getMatrix(city, nextCity)

						# if we have a completed tour
						if len(potential[4]) == 0:
							solutions += 1
							tour = potential[5]
							# O(n) time and O(n) space
							temp = TSPSolution([self.cities[i] for i in tour])
							# if the cost is less than the current best, update the best
							if temp.cost < self.lowestBound:
								self.lowestBound = temp.cost
								bestsFound += 1
								bssf = temp

						# if we have not completed the tour, add the city to the heap
						else:
							if potential[0] < self.lowestBound:
								# lower than bound, so continue exploring
								heapq.heappush(heap, potential)
								queueSize = max(queueSize, len(heap))
								totalStates += 1
							else:
								# higher than bound, so prune
								pruned += 1
								totalStates += 1
			queueSize = max(queueSize, len(heap))
		
		end = time.time()
		results = {}
		results['cost'] = self.lowestBound
		results['time'] = end - start
		results['count'] = solutions
		results['soln'] = bssf
		results['max'] = queueSize
		results['total'] = totalStates
		results['pruned'] = pruned
		return results



	# O(n^2) time and space
	def getBaseMatrix(self, cities):
		matrix = np.full((self.numCities, self.numCities), np.inf)

		# initialize matrix (O(n^2))
		for i in range(self.numCities):
			for j in range(self.numCities):
				if i != j:
					matrix[i][j] = cities[i].costTo(cities[j])

		lowestBound = 0

		# reduce rows (O(n))
		for i in range(self.numCities):
			lowestBound += np.min(matrix[i])
			matrix[i] -= np.min(matrix[i])

		# reduce columns (O(n))
		for i in range(self.numCities):
			lowestBound += np.min(matrix[:, i])
			matrix[:, i] -= np.min(matrix[:, i])

		return matrix, lowestBound

	# O(n) time and O(n^2) space
	def getMatrix(self, city, givenTuple):
		# copy current partial path tuple
		tupleCopy = deepcopy(givenTuple)
		matrix = tupleCopy[2]
		initialCost = matrix[tupleCopy[3]._index][city._index]
		reducedSum = 0

		# set row and column of city to infinity
		matrix[tupleCopy[3]._index] = np.inf
		matrix[:, city._index] = np.inf
		matrix[city._index][tupleCopy[3]._index] = np.inf

		# reduce rows (O(n) time)
		for row in range(matrix.shape[0]):
			rowMin = np.min(matrix[row])
			if rowMin == np.inf:
				continue
			matrix[row][city._index] -= rowMin
			reducedSum += rowMin

		# reduce columns (O(n) time)
		for col in range(matrix.shape[1]):
			colMin = np.min(matrix[:, col])
			if colMin == np.inf:
				continue
			matrix[:, col] -= colMin
			reducedSum += colMin
		
		# remove visited city from remaining cities		
		# (O(n) time)
		remainingCities = tupleCopy[4]
		remainingCities = [c for c in remainingCities if c._index != city._index]

		# get new cost
		cost = tupleCopy[0] + initialCost + reducedSum

		# return tuple for partial path generated from visiting city
		return (cost, next(tiebreaker), matrix, city, remainingCities, tupleCopy[5] + [city._index])


	# get MST using Prim's algorithm
	# O(E logN) time and O(E) space, where E is the number of edges and N is the number of cities
	def getMST(cities):
		# MST is an array of tuples (city1, city2, cost)
		mst = []
		numCities = len(cities)
		visited = [False] * numCities
		visited[0] = True
		numEdges = 0

		# in minimum spanning tree, there are n - 1 edges
		while numEdges < numCities - 1:
			minEdge = np.inf
			minCity = None
			minIndex = None

			for i in range(numCities):
				if visited[i]:
					for j in range(numCities):
						if not visited[j] and cities[i].costTo(cities[j]) < minEdge:
							minEdge = cities[i].costTo(cities[j])
							minCity = cities[j]
							minIndex = j

			mst.append(tuple((cities[i], minCity, minEdge)))
			visited[minIndex] = True
			numEdges += 1

		return mst