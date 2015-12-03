import numpy as np

# read in text file
textFile = open( 'The_Innocents_Abroad.txt', 'r' ).read( )
characters = { chr( x ) for x in range( 0, 256 ) }
numberOfCharacters = 256
characterToIndex = { character : index for index, character in enumerate( characters ) }
indexToCharacter = { index : character for index, character in enumerate( characters ) }

# define network architecture
hiddenLayerSize = 100
timeSteps = 25
learningRate = 1e-1
inputToHiddenWeights = np.random.randn( hiddenLayerSize, numberOfCharacters ) * 0.01
hiddenToHiddenWeights = np.random.randn( hiddenLayerSize, hiddenLayerSize ) * 0.01
hiddenToOutputWeights = np.random.randn( numberOfCharacters, hiddenLayerSize ) * 0.01
hiddenBias = np.zeros( ( hiddenLayerSize, 1 ) )
outputBias = np.zeros( ( numberOfCharacters, 1 ) )

def lossFunction( inputs, targets, previousHiddenState ):
  """
  Inputs: inputs and targets are both lists of integers;
          previousHiddenState is an Hx1 array of the initial hidden state
  Return: the loss, gradients on model parameters, and the last hidden state
  """
  inputOverTime, hiddenStates, outputOverTime, predictionsOverTime = { }, { }, { }, { }
  hiddenStates[ -1 ] = np.copy( previousHiddenState )
  loss = 0
  temperature = 1
  # forward pass
  for t in xrange( len( inputs ) ):
    inputOverTime[ t ] = np.zeros( ( numberOfCharacters,1 ) ) # encode in 1-of-k representation
    inputOverTime[ t ][ inputs[ t ] ] = 1
    hiddenStates[ t ] = np.tanh( np.dot( inputToHiddenWeights, inputOverTime[ t ] ) + np.dot( hiddenToHiddenWeights, \
    hiddenStates[ t-1 ] ) + hiddenBias )
    outputOverTime[ t ] = np.dot( hiddenToOutputWeights, hiddenStates[ t ] ) + outputBias # unnormalized log probabilities for next chars
    predictionsOverTime[ t ] = (np.exp( outputOverTime[ t ]) / temperature) / np.sum( np.exp( outputOverTime[ t ] ) / temperature) # probabilities for next chars
    loss += -np.log( predictionsOverTime[ t ][ targets[ t ], 0 ] ) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like( inputToHiddenWeights ), np.zeros_like( hiddenToHiddenWeights ), \
  np.zeros_like( hiddenToOutputWeights )
  dbh, dby = np.zeros_like( hiddenBias ), np.zeros_like( outputBias )
  dhnext = np.zeros_like( hiddenStates[ 0 ] )
  for t in reversed( xrange( len( inputs ) ) ):
    dy = np.copy( predictionsOverTime[ t ] )
     # backprop into y
    dy[ targets[ t ] ] -= 1
    dWhy += np.dot( dy, hiddenStates[ t ].T )
    dby += dy
     # backprop into h
    dh = np.dot( hiddenToOutputWeights.T, dy ) + dhnext
    # tanh backprop
    dhraw = ( 1 - hiddenStates[ t ] * hiddenStates[ t ] ) * dh
    dbh += dhraw
    dWxh += np.dot( dhraw, inputOverTime[ t ].T )
    dWhh += np.dot( dhraw, hiddenStates[ t - 1 ].T )
    dhnext = np.dot( hiddenToHiddenWeights.T, dhraw )
  for dparam in [ dWxh, dWhh, dWhy, dbh, dby ]:
    # clip to mitigate "exploding" gradients
    np.clip(dparam, -5, 5, out=dparam)
  return loss, dWxh, dWhh, dWhy, dbh, dby, hiddenStates[ len( inputs ) - 1 ]

def sampleFromFile( hiddenState, seedIndex, sampleLength ):
  """ 
  Samples a sequence of integers from the model
  hiddenState is the memory state, seedIndex is seed letter for first time step
  """
  x = np.zeros( ( numberOfCharacters, 1 ) )
  x[ seedIndex ] = 1
  indicies = [ ]
  for t in xrange( sampleLength ):
    hiddenState = np.tanh( np.dot( inputToHiddenWeights, x ) + np.dot( hiddenToHiddenWeights, hiddenState ) \
    + hiddenBias )
    output = np.dot( hiddenToOutputWeights, hiddenState ) + outputBias
    probabilites = np.exp( output ) / np.sum( np.exp( output ) )
    index = np.random.choice( range( numberOfCharacters ), probabilities = probabilites.ravel( ) )
    x = np.zeros( ( numberOfCharacters, 1 ) )
    x[ index ] = 1
    indicies.append( index )
  return indicies

totalIterations, fileIndex = 0, 0
 # memory variables for Adagrad
mWxh, mWhh, mWhy = np.zeros_like( inputToHiddenWeights ), np.zeros_like( hiddenToHiddenWeights ), \
np.zeros_like( hiddenToOutputWeights )
mbh, mby = np.zeros_like( hiddenBias ), np.zeros_like( outputBias )
 # loss before first iteration
smoothLoss = -np.log( 1.0 / numberOfCharacters ) * timeSteps
lossPerEpoch = [ ]
sampleOutput = [ ]
maxEpochs = 5
while len( sampleOutput ) < maxEpochs:
  if fileIndex + timeSteps + 1 >= len( textFile ) or totalIterations == 0:
    # reset RNN memory
    previousHiddenState = np.zeros( ( hiddenLayerSize, 1 ) )
    # go from start of data
    fileIndex = 0
    lossPerEpoch.append( smoothLoss )
  inputs = [ characterToIndex[ ch ] for ch in textFile[ fileIndex : fileIndex + timeSteps ] ]
  targets = [ characterToIndex[ ch ] for ch in textFile[ fileIndex + 1 : fileIndex + timeSteps + 1 ] ]

  # sample from the model after every epoch
  if fileIndex == 0 and totalIterations > 1:
    sampleIndicies = sampleFromFile( previousHiddenState, inputs[ 0 ], 100 )
    text = ''.join( indexToCharacter[ ix ] for ix in sampleIndicies )
    sampleOutput.append( text )
    print '\niter %d, number of samples: %d \n' % ( totalIterations, len( sampleOutput ) )
    print 'txt %d: %s' % ( len( sampleOutput ), text[0:99] )
    
  # calculate gradients
  loss, dWxh, dWhh, dWhy, dbh, dby, previousHiddenState = lossFunction( inputs, targets, previousHiddenState )
  smoothLoss = smoothLoss * 0.999 + loss * 0.001
  
  # Adagrad
  for param, dparam, memory in zip([ inputToHiddenWeights, hiddenToHiddenWeights, \
  hiddenToOutputWeights, hiddenBias, outputBias, ], [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    memory += dparam * dparam
    param += -learningRate * dparam / np.sqrt( memory + 1e-8 )
  
  # move to next sequence of characters to read in
  fileIndex += timeSteps
  totalIterations += 1