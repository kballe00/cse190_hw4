import numpy as np

# read in text file
textFile = open( 'The Innocents Abroad.txt', 'r' ).read( )
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

def lossFunction( inputs, targets, hprev ):
  """
  Inputs: inputs ad targets are both list of integers;
          hprev is an Hx1 array of the initial hidden state
  Return: the loss, gradients on model parameters, and the last hidden state
  """
  inputOverTime, hiddenStates, outputOverTime, predictionsOverTime = { }, { }, { }, { }
  hiddenStates[ -1 ] = np.copy( hprev )
  loss = 0
  # forward pass
  for t in xrange( len( inputs ) ):
    inputOverTime[ t ] = np.zeros( ( numberOfCharacters,1 ) ) # encode in 1-of-k representation
    inputOverTime[ t ][ inputs[ t ] ] = 1
    hiddenStates[ t ] = np.tanh( np.dot( inputToHiddenWeights, inputOverTime[ t ] ) + np.dot( hiddenToHiddenWeights, \
    hiddenStates[ t-1 ] ) + hiddenBias )
    outputOverTime[ t ] = np.dot( hiddenToOutputWeights, hiddenStates[ t ] ) + outputBias # unnormalized log probabilities for next chars
    predictionsOverTime[ t ] = np.exp( outputOverTime[ t ]) / np.sum( np.exp( outputOverTime[ t ] ) ) # probabilities for next chars
    loss += -np.log( predictionsOverTime[ t ][ targets[ t ], 0 ] ) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like( inputToHiddenWeights ), np.zeros_like( hiddenToHiddenWeights ), \
  np.zeros_like( hiddenToOutputWeights )
  dbh, dby = np.zeros_like( hiddenBias ), np.zeros_like( outputBias )
  dhnext = np.zeros_like( hiddenStates[ 0 ] )
  for t in reversed( xrange( len( inputs ) ) ):
    dy = np.copy( predictionsOverTime[ t ] )
    dy[ targets[ t ] ] -= 1 # backprop into y
    dWhy += np.dot( dy, hiddenStates[ t ].T )
    dby += dy
    dh = np.dot( hiddenToOutputWeights.T, dy ) + dhnext # backprop into h
    dhraw = ( 1 - hiddenStates[ t ] * hiddenStates[ t ] ) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot( dhraw, inputOverTime[ t ].T )
    dWhh += np.dot( dhraw, hiddenStates[ t - 1 ].T )
    dhnext = np.dot( hiddenToHiddenWeights.T, dhraw )
  for dparam in [ dWxh, dWhh, dWhy, dbh, dby ]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hiddenStates[ len( inputs ) - 1 ]

def sample( h, seed_ix, n ):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros( ( numberOfCharacters, 1 ) )
  x[ seed_ix ] = 1
  ixes = [ ]
  for t in xrange( n ):
    h = np.tanh( np.dot( inputToHiddenWeights, x ) + np.dot( hiddenToHiddenWeights, h ) \
    + hiddenBias )
    y = np.dot( hiddenToOutputWeights, h ) + outputBias
    p = np.exp( y ) / np.sum( np.exp( y ) )
    ix = np.random.choice( range( numberOfCharacters ), p = p.ravel( ) )
    x = np.zeros( ( numberOfCharacters, 1 ) )
    x[ ix ] = 1
    ixes.append( ix )
  return ixes

totalIterations, fileIndex = 0, 0
mWxh, mWhh, mWhy = np.zeros_like( inputToHiddenWeights ), np.zeros_like( hiddenToHiddenWeights ), \
np.zeros_like( hiddenToOutputWeights )
mbh, mby = np.zeros_like( hiddenBias ), np.zeros_like( outputBias ) # memory variables for Adagrad
smooth_loss = -np.log( 1.0 / numberOfCharacters ) * timeSteps # loss at iteration 0
lossPerEpoch = [ ]
sampleOutput = [ ]
maxEpochs = 5
while len( sampleOutput ) < maxEpochs:
  if fileIndex + timeSteps + 1 >= len( textFile ) or totalIterations == 0: 
    hprev = np.zeros( ( hiddenLayerSize, 1 ) ) # reset RNN memory
    fileIndex = 0 # go from start of data
    lossPerEpoch.append( smooth_loss )
  inputs = [ characterToIndex[ ch ] for ch in textFile[ fileIndex : fileIndex + timeSteps ] ]
  targets = [ characterToIndex[ ch ] for ch in textFile[ fileIndex + 1 : fileIndex + timeSteps + 1 ] ]

  # sample from the model after every epoch
  if fileIndex == 0 and totalIterations > 1:
    sample_ix = sample(hprev, inputs[ 0 ], 200)
    txt = ''.join( indexToCharacter[ ix ] for ix in sample_ix )
    sampleOutput.append( txt )
    print 'iter %d, number of samples: %d' % ( totalIterations, len( sampleOutput ) )
    
  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFunction(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  
  # perform parameter update with Adagrad
  for param, dparam, memory in zip([ inputToHiddenWeights, hiddenToHiddenWeights, \
  hiddenToOutputWeights, hiddenBias, outputBias, ], [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    memory += dparam * dparam
    param += -learningRate * dparam / np.sqrt( memory + 1e-8 ) # adagrad update

  # move to next sequence of characters to read in
  fileIndex += timeSteps
  totalIterations += 1