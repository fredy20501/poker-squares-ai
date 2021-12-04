import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.HashMap;
import java.util.List;
import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

/**
 * (Hand Abstraction Reinforcement Greedy Probability Monte Carlo Player)
 * HARGPMCPlayer - a Monte Carlo implementation of the player interface for PokerSquares using an 
 * optimized greedy search as a playout policy, hand abstractions & probabilities for partial evaluation, 
 * and reinforcement learning to learn hand abstraction values.
 */
public class HARGPMCPlayer implements PokerSquaresPlayer {
	
	private final int SIZE = 5; // number of rows/columns in square grid
	private final int NUM_POS = SIZE * SIZE; // number of positions in square grid
	private final int NUM_CARDS = Card.NUM_CARDS; // number of cards in deck
	private Random random = new Random(); // pseudorandom number generator for Monte Carlo simulation 
	private int[] plays = new int[NUM_POS]; // positions of plays so far (index 0 through numPlays - 1) recorded as integers using row-major indices.
											// row-major indices: play (r, c) is recorded as a single integer r * SIZE + c (See http://en.wikipedia.org/wiki/Row-major_order)
											// From plays index [numPlays] onward, we maintain a list of yet unplayed positions.
	private int numPlays = 0; // number of Cards played into the grid so far
	private PokerSquaresPointSystem system; // point system
	private int depthLimit = 25; // default depth limit for Monte Carlo (MC) play
	private Card[][] grid = new Card[SIZE][SIZE]; // grid with Card objects or null (for empty positions)
	private Card[] simDeck = Card.getAllCards(); // a list of all Cards. As we learn the index of cards in the play deck,
	                                             // we swap each dealt card to its correct index.  Thus, from index numPlays 
												 // onward, we maintain a list of undealt cards for MC simulation.
	private int[][] legalPlayLists = new int[NUM_POS][NUM_POS]; // stores legal play lists indexed by numPlays (depth)
	// (This avoids constant allocation/deallocation of such lists during the selections of MC simulations.)

	// New fields (different from RandomMCPlayer)
	private HashMap<Integer,Float[]> abstractionUtilities; // Hashmap storing the average utility for hand abstractions.
	private File utilityFile;
	public float epsilon = 0.5f; // Initial probability of making a random move during Monte Carlo simulation
	private int[][] trainingAbstractions = new int[SIZE * 2][SIZE-1]; // Stores the 4 partial hand abstractions that occur during the game 
																	  // for each of the 10 rows/cols (only used if training)

	public boolean isTraining = false;

	/**
	 * Create a Random Monte Carlo player that simulates random play to depth 2.
	 */
	public HARGPMCPlayer() {}
	
	/**
	 * Create a Random Monte Carlo player that simulates random play to a given depth limit.
	 * @param depthLimit depth limit for random simulated play
	 */
	public HARGPMCPlayer(int depthLimit) {
		this.depthLimit = depthLimit;
	}
	
	/* (non-Javadoc)
	 * @see PokerSquaresPlayer#init()
	 */
	@Override
	public void init() { 
		utilityFile = new File(getName()+"_HAUtilities.map");
		loadUtilityMap();
		// clear grid
		for (int row = 0; row < SIZE; row++)
			for (int col = 0; col < SIZE; col++)
				grid[row][col] = null;
		// reset numPlays
		numPlays = 0;
		// (re)initialize list of play positions (row-major ordering)
		for (int i = 0; i < NUM_POS; i++)
			plays[i] = i;
	}

	/* (non-Javadoc)
	 * @see PokerSquaresPlayer#getPlay(Card, long)
	 */
	@Override
	public int[] getPlay(Card card, long millisRemaining) {
		/*
		 * With this algorithm, the player chooses the legal play that has the highest expected score outcome.
		 * This outcome is estimated as follows:
		 *   For each move, many simulated random plays to the set depthLimit are performed and the (sometimes
		 *     partially-filled) grid is scored.
		 *   For each play simulation, random undrawn cards are drawn in simulation and the player
		 *     picks a play position randomly.
		 *   After many such plays, the average score per simulated play is computed.  The play with the highest 
		 *     average score is chosen (breaking ties randomly).   
		 */
		
		// match simDeck to actual play event; in this way, all indices forward from the card contain a list of 
		//   undealt Cards in some permutation.
		int cardIndex = numPlays;
		while (!card.equals(simDeck[cardIndex]))
			cardIndex++;
		simDeck[cardIndex] = simDeck[numPlays];
		simDeck[numPlays] = card;

		if (numPlays < 24) { // not the forced last play
			// compute average time per move evaluation
			int remainingPlays = NUM_POS - numPlays; // ignores triviality of last play to keep a conservative margin for game completion
			long millisPerPlay = millisRemaining / remainingPlays; // dividing time evenly with future getPlay() calls
			long millisPerMoveEval = millisPerPlay / remainingPlays; // dividing time evenly across moves now considered
			// copy the play positions (row-major indices) that are empty
			System.arraycopy(plays, numPlays, legalPlayLists[numPlays], 0, remainingPlays);
			float maxAverageScore = Float.NEGATIVE_INFINITY; // maximum average score found for moves so far
			ArrayList<Integer> bestPlays = new ArrayList<Integer>(); // all plays yielding the maximum average score 
			for (int i = 0; i < remainingPlays; i++) { // for each legal play position
				int play = legalPlayLists[numPlays][i];
				long startTime = System.currentTimeMillis();
				long endTime = startTime + millisPerMoveEval; // compute when MC simulations should end
				makePlay(card, play / SIZE, play % SIZE);  // play the card at the empty position
				int simCount = 0;
				float scoreTotal = 0;
				while (System.currentTimeMillis() < endTime) { // perform as many MC simulations as possible through the allotted time
					// Perform a Monte Carlo simulation of random play to the depth limit or game end, whichever comes first.
					scoreTotal += simPlay(depthLimit);  // accumulate MC simulation scores
					simCount++; // increment count of MC simulations
				}
				undoPlay(); // undo the play under evaluation
				// update (if necessary) the maximum average score and the list of best plays
				float averageScore = scoreTotal / simCount;
				if (averageScore >= maxAverageScore) {
					if (averageScore > maxAverageScore)
						bestPlays.clear();
					bestPlays.add(play);
					maxAverageScore = averageScore;
				}
			}
			int bestPlay = bestPlays.get(random.nextInt(bestPlays.size())); // choose a best play (breaking ties randomly)
			// update our list of plays, recording the chosen play in its sequential position; all onward from numPlays are empty positions
			int bestPlayIndex = numPlays;
			while (plays[bestPlayIndex] != bestPlay)
				bestPlayIndex++;
			plays[bestPlayIndex] = plays[numPlays];
			plays[numPlays] = bestPlay;
		}

		// decode play into row and column
		int row = plays[numPlays] / SIZE;
		int col = plays[numPlays] % SIZE;
		makePlay(card, row, col); // make the chosen play (not undoing this time)
		
		// (only for training) Keep track of partial abstraction as the game progress 
		if (isTraining) {
			// Get hands affected by the new play
			Card[] rowHand = getHandByRow(row);
			Card[] colHand = getHandByColumn(col);
			
			/* Save partial hand abstractions (1 to 4 card hands) for both row and column */
			// Row
			int numCards = 0;
			for (Card c : rowHand) {
				if (c != null) numCards++;
			}
			if (numCards > 0 && numCards < SIZE) {
				trainingAbstractions[row][numCards-1] = getHandAbstraction(rowHand, true);
			}
			
			// Col
			numCards = 0;
			for (Card c : colHand) {
				if (c != null) numCards++;
			}
			if (numCards > 0 && numCards < SIZE) {
				trainingAbstractions[col+SIZE][numCards-1] = getHandAbstraction(colHand, false);
			}

			// After the last play, update the utility map with the final hand score for each row/col
			if (numPlays==25) {
				int[] handScores = system.getHandScores(grid);
				for (int i = 0; i < handScores.length; i++) {
					int handScore = handScores[i];
					for (int j = 0; j < trainingAbstractions[i].length; j++) {
						int abstraction = trainingAbstractions[i][j];
						if (abstractionUtilities.containsKey(abstraction)) {
							// Update entry
							Float[] utilityArray = abstractionUtilities.get(abstraction);
							Float prevMean = utilityArray[0];
							Float count = utilityArray[1]+1.0f;
							utilityArray[0] = prevMean + (handScore - prevMean)/count;
							utilityArray[1] = count;
							abstractionUtilities.put(abstraction, utilityArray);
						}
						else {
							// Create entry
							Float[] utilityArray = {(float)handScore, 1.0f};
							abstractionUtilities.put(abstraction, utilityArray);
						}
					}
				}
				// Save the updated utility map
				saveUtilityMap();
			}
		}

		// Return the chosen play
		int[] playPos = {row, col};
		return playPos; 
	}

	/**
	 * From the chosen play, perform simulated Card draws and random placement (depthLimit) iterations forward 
	 * and return the resulting grid score.
	 * @param depthLimit - how many simulated random plays to perform
	 * @return resulting grid score after random MC simulation to given depthLimit
	 */
	private float simPlay(int depthLimit) {
		if (depthLimit == 0) { // with zero depth limit, return current score
			return getGridUtility();
		}
		else { // up to the non-zero depth limit or to game end, iteratively make the given number of random plays
			int depth = Math.min(depthLimit, NUM_POS - numPlays); // compute real depth limit, taking into account game end
			for (int d = 0; d < depth; d++) {
				// generate a random card draw
				int c = random.nextInt(NUM_CARDS - numPlays) + numPlays;
				Card card = simDeck[c];

				// Get list of remaning legal plays
				int remainingPlays = NUM_POS - numPlays;
				System.arraycopy(plays, numPlays, legalPlayLists[numPlays], 0, remainingPlays);

				// Short-circuit the forced last play
				if (remainingPlays == 1) {
					int play = legalPlayLists[numPlays][0];
					makePlay(card, play / SIZE, play % SIZE);
					break;
				}

				// Choose play that gives best utility increase
				float maxUtility = Integer.MIN_VALUE;
				ArrayList<Integer> bestPlays = new ArrayList<Integer>(); // all plays yielding the maximum utility
				for (int i = 0; i < remainingPlays; i++) {
					int currentPlay = legalPlayLists[numPlays][i];
					// Calculate utility for the current row+col of this play
					int row = currentPlay / SIZE;
					int col = currentPlay % SIZE;
					Card[] rowHand = getHandByRow(row);
					Card[] columnHand = getHandByRow(col);
					float currentUtility = getHandUtility(rowHand, true)+getHandUtility(columnHand, false);
					// Calculate utility for the row+col if the play was made
					rowHand[col] = card;
					columnHand[row] = card;
					float afterPlayUtility = getHandUtility(rowHand, true)+getHandUtility(columnHand, false);
					// Keep track of max utility increase
					float playUtility = afterPlayUtility - currentUtility;
					if (playUtility >= maxUtility) {
						if (playUtility > maxUtility) {
							bestPlays.clear();
						}
						bestPlays.add(currentPlay);
						maxUtility = playUtility;
					}
				}
				int play = -1;
				// choose a best play (breaking ties randomly)
				if (bestPlays.size() > 0) play = bestPlays.get(random.nextInt(bestPlays.size()));

				if (isTraining) {
					// Use random play with probability P=epsilon
					float x = random.nextFloat();
					if (x<epsilon) play = -1;
				}

				if (play == -1) {
					// Choose a random play from the remaining legal plays
					int c2 = random.nextInt(remainingPlays);
					play = legalPlayLists[numPlays][c2];
				}

				makePlay(card, play / SIZE, play % SIZE);
			}
			float score = getGridUtility();

			// Undo MC plays.
			for (int d = 0; d < depth; d++) {
				undoPlay();
			}

			return score;
		}
	}
	
	public void makePlay(Card card, int row, int col) {
		// match simDeck to event
		int cardIndex = numPlays;
		while (!card.equals(simDeck[cardIndex]))
			cardIndex++;
		simDeck[cardIndex] = simDeck[numPlays];
		simDeck[numPlays] = card;
		
		// update plays to reflect chosen play in sequence
		grid[row][col] = card;
		int play = row * SIZE + col;
		int j = 0;
		while (plays[j] != play)
			j++;
		plays[j] = plays[numPlays];
		plays[numPlays] = play;
		
		// increment the number of plays taken
		numPlays++;
	}

	public void undoPlay() { // undo the previous play
		numPlays--;
		int play = plays[numPlays];
		grid[play / SIZE][play % SIZE] = null;	
	}

	/* (non-Javadoc)
	 * @see PokerSquaresPlayer#setPointSystem(PokerSquaresPointSystem, long)
	 */
	@Override
	public void setPointSystem(PokerSquaresPointSystem system, long millis) {
		this.system = system;
	}

	/* (non-Javadoc)
	 * @see PokerSquaresPlayer#getName()
	 */
	@Override
	public String getName() {
		return "HARGPMCPlayerDepth" + depthLimit;
	}

	@SuppressWarnings("unchecked")
	private void loadUtilityMap() {
		// Check if file exists
		if (!utilityFile.isFile() && !isTraining) {
			// Show error to user
			System.out.println("Error (Player "+getName()+") - could not find utility map file "+utilityFile.getName()+". Make sure the file exists and is in the same directory.");
			abstractionUtilities = new HashMap<Integer,Float[]>();
			return;
		}

		try (
			FileInputStream f = new FileInputStream(utilityFile);
			ObjectInputStream s = new ObjectInputStream(f);
		) {
			abstractionUtilities = (HashMap<Integer,Float[]>)s.readObject();
		}
		catch (Exception e) {
			// Can't load file - initialize to empty map
			abstractionUtilities = new HashMap<Integer,Float[]>();
			if (!isTraining) {
				System.out.println("Error (Player "+getName()+") - could not load utility map from file "+utilityFile.getName()+". Message: "+e.getMessage());
			}
		}
	}

	private void saveUtilityMap() {
		// This should only be used when training
		if (!isTraining) return;

		// Create file if doesn't exist
		if (!utilityFile.isFile() && !isTraining) {
			System.out.print("Utility map file doesn't exist. Creating new one...");
			try {
				utilityFile.createNewFile();
				System.out.println("Done!");
			}
			catch (Exception e) {
				System.out.println("Error: "+e.getMessage());
			}
		}

		// Save to file (override if exists)
		try (
			FileOutputStream f = new FileOutputStream(utilityFile);
			ObjectOutputStream s = new ObjectOutputStream(f);
		) {
			s.writeObject(abstractionUtilities);
		}
		catch (Exception e) {
			System.out.println("Error (Player "+getName()+") - could not save utility map to file "+utilityFile.getName()+". Message: "+e.getMessage());
		}
	}

	// Used for debugging
	public void printUtilityMap() {
		System.out.println("[Utility Map]");
		abstractionUtilities.entrySet().forEach(entry -> {
			Float[] utilityArray = entry.getValue();
			String abstraction = abstractionToString(entry.getKey());
			System.out.println(abstraction + ": avg(" +utilityArray[0]+") count(" + utilityArray[1]+")");
		});
		System.out.println();
	}

	private float getGridUtility() {
		float gridUtility = 0;
		for (int row = 0; row < SIZE; row++) {
			gridUtility += getHandUtility(getHandByRow(row), true);
		}
		for (int col = 0; col < SIZE; col++) {
			gridUtility += getHandUtility(getHandByColumn(col), false);
		}
		return gridUtility;
	}

	private Card[] getHandByRow(int row) {
		Card[] hand = new Card[SIZE];
		for (int col = 0; col < SIZE; col++) {
			hand[col] = grid[row][col];
		}
		return hand;
	}

	private Card[] getHandByColumn(int col) {
		Card[] hand = new Card[SIZE];
		for (int row = 0; row < SIZE; row++) {
			hand[row] = grid[row][col];
		}
		return hand;
	}

	private float getHandUtility(Card[] hand, boolean isRow) {
		float utility = 0;
		int numCards = 0;
		for (int i = 0; i < SIZE ; i++) {
			if (hand[i] != null) numCards++;
		}
		if (numCards == 0) utility = 0;
		else if (numCards == 5) utility = system.getHandScore(hand);
		else if (numCards == 4) utility = getExpectedValue(hand);
		else utility = getPartialHandUtility(hand, isRow);
		return utility;
	}

	public float getExpectedValue(Card[] hand) {
		float expectedValue = 0;
		int [] suitArr  = new int[4];
		int [] rankArr  = new int[13];

		for (int i = 0; i < 5; i++) {
			if (hand[i] != null) {
				suitArr[hand[i].getSuit()]++;
				rankArr[hand[i].getRank()]++;
			}
		}
		List<Integer> suitArrayList = Arrays.stream(suitArr).boxed().collect(Collectors.toList());
		List<Integer> rankArrayList = Arrays.stream(rankArr).boxed().collect(Collectors.toList());
		float [] probArray = new float [9];

		if (suitArrayList.contains(4)) {
			probArray[0] = (probOfRoyalFlush(hand, suitArrayList.indexOf(4))); // royal flush
			probArray[6] = probOfSuit(suitArrayList.indexOf(4)); // flush
		}
		if (rankArrayList.contains(4)) {
			probArray[2] = 1; // fourOfKind
		}
		else if (rankArrayList.contains(3)) {
			probArray[5] = 1; // threeOfKind
			probArray[2] = probOfRank(rankArrayList.indexOf(3)); //fourOfKind
			probArray[4] = probOfRank(rankArrayList.indexOf(1)); // find card to complete 2 card rank for full-house
		}
		else if (rankArrayList.contains(2)) {
			probArray[8] = 1; // 1 pair
			if (Collections.frequency(rankArrayList, 2) == 2) { // check if there's two occurence of 2 
				probArray[7] = 1; // 2 pair
				probArray[4] = probOfRank(rankArrayList.indexOf(2), rankArrayList.lastIndexOf(2)); // find card for 3 card rank of full-house
				probArray[5] = probArray[4]; // threeOfKind
			}
			else {
				probArray[5] = probOfRank(rankArrayList.indexOf(2)); //threeOfKind
				probArray[7] = probOfRank(rankArrayList.indexOf(1), rankArrayList.lastIndexOf(1)); // twoPair
			}
		}
		else {
			probArray[3] = probOfSequence(hand, -1); // straight
			probArray[8] = probOfRank(rankArrayList); // one pair
			if (suitArrayList.contains(4)) {
				probArray[1] = probOfSequence(hand, suitArrayList.indexOf(4)); // straight flush 
			}
		}  
		
		float p;
		int [] utilityArray = {
			system.getHandScore(PokerHand.ROYAL_FLUSH),
			system.getHandScore(PokerHand.STRAIGHT_FLUSH),
			system.getHandScore(PokerHand.FOUR_OF_A_KIND),
			system.getHandScore(PokerHand.STRAIGHT),
			system.getHandScore(PokerHand.FULL_HOUSE),
			system.getHandScore(PokerHand.THREE_OF_A_KIND),
			system.getHandScore(PokerHand.FLUSH),
			system.getHandScore(PokerHand.TWO_PAIR),
			system.getHandScore(PokerHand.ONE_PAIR),
		};

		// String [] utilityNameArray = {
		// 	"ROYAL_FLUSH",
		// 	"STRAIGHT_FLUSH",
		// 	"FOUR_OF_A_KIND",
		// 	"STRAIGHT",
		// 	"FULL_HOUSE",
		// 	"THREE_OF_A_KIND",
		// 	"FLUSH",
		// 	"TWO_PAIR",
		// 	"ONE_PAIR",
		// };

		for (int i=0; i < 9; i++) {
			p = probArray[i];
			for (int j=0; j < i; j++) {
				 p *= (1-probArray[j]);
			}
			expectedValue += p * utilityArray[i];
		}

		return expectedValue;
	}

	float probOfRoyalFlush(Card[] hand, int suit) {
		int rankToGet = 0;
		boolean correctSequence = true;
		ArrayList<Integer> royalFlush = new ArrayList<Integer>(Arrays.asList(0, 9, 10, 11, 12));
		
		for (int i=0; i < 5; i++) {
			if (hand[i] != null && royalFlush.contains(hand[i].getRank())) {
				royalFlush.remove((Integer) hand[i].getRank());
			}
			else if(hand[i] != null) {
				correctSequence = false;
			}
		}
		
		int undealtRoyalCardCount = 0;
		if (correctSequence) { // probability of missing rank with specified suit
			rankToGet = royalFlush.get(0);

			for (int i=numPlays; i<simDeck.length; i++) {
				Card card = simDeck[i];
				if (card.getSuit() == suit && card.getRank() == rankToGet) undealtRoyalCardCount++;
			}
			return undealtRoyalCardCount/((float)(NUM_CARDS-numPlays));
		}
		else {
			return 0;
		}
	}
	
	float probOfSequence(Card[] hand, int suit) {
		ArrayList <Integer> rankList = new ArrayList<>();
		int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
		for (int i=0; i<5; i++) {
			if (hand[i] != null) {
				if (hand[i].getRank() > max) max = hand[i].getRank();
				if (hand[i].getRank() < min) min = hand[i].getRank();
				rankList.add(hand[i].getRank());
			}
		}

		int diff = max - min;
		ArrayList<Integer> ranksToGet = new ArrayList<>();
		if (diff <= 4) {
			if (diff == 4) { // missing card is between min-max
				for (int i = min; i < max; i++) {
					if (!rankList.contains(i)) {
						ranksToGet.add(i);
					}
				}				
			}
			else {
				if (max != 12) {
					ranksToGet.add(max + 1);
				}
				if (min != 0) {
					ranksToGet.add(min - 1);
				}
			}
		}
		else {
			return 0;
		}

		int eligibleCards = 0;
		for (int i=numPlays; i<simDeck.length; i++) {
			Card card = simDeck[i];
			if (ranksToGet.contains(card.getRank())  && ( suit == -1 || card.getSuit() == suit)) {
				eligibleCards++;
			} 
		}
		return eligibleCards/((float)(NUM_CARDS-numPlays));
	}

	float probOfSuit(int suitIndex) {
		int undealtSuitCount = 0;
		for (int i=numPlays; i<simDeck.length; i++) {
			Card card = simDeck[i];
			if (card.getSuit() == suitIndex) undealtSuitCount++;
		}
		return (undealtSuitCount/((float)(NUM_CARDS-numPlays)));
	}

	float probOfRank(int rankIndex) {
		int undealtRankCount = 0;
		for (int i=numPlays; i<simDeck.length; i++) {
			Card card = simDeck[i];
			if (card.getRank() == rankIndex) undealtRankCount++;
		}
		return (undealtRankCount/((float)(NUM_CARDS-numPlays)));
	}

	float probOfRank(int rankIndex1, int rankIndex2) {
		int undealtRankCount = 0;
		for (int i=numPlays; i<simDeck.length; i++) {
			Card card = simDeck[i];
			if (card.getRank() == rankIndex1) undealtRankCount++;
			else if(card.getRank() == rankIndex2) undealtRankCount++;
		}
		return (undealtRankCount/((float)(NUM_CARDS-numPlays)));
	}

	float probOfRank(List<Integer> rankArrayList) {
		int undealtRankCount = 0;
		for (int i=numPlays; i<simDeck.length; i++) {
			Card card = simDeck[i];
			if (rankArrayList.get(card.getRank()) > 0) undealtRankCount++;
		}
		return (undealtRankCount/((float)(NUM_CARDS-numPlays)));
	}

	private float getPartialHandUtility(Card[] hand, boolean isRow) {
		int abstraction = getHandAbstraction(hand, isRow);
		float utility = evaluateHandAbstraction(abstraction);
		return utility;
	}

	private float evaluateHandAbstraction(int abstraction) {
		// Get the utility from the map (0 if doesn't exist)
		float utility = 
			abstractionUtilities.containsKey(abstraction) ? 
			abstractionUtilities.get(abstraction)[0] : 0;
		return utility;
	}

	// Used for partial hands of 1 to 4 cards (undefined for 0 or 5 card hands)
	private int getHandAbstraction(Card[] hand, boolean isRow) {
		/**
		 * 16 bit Abstraction
		 * 3 bits: number of cards without a pair
		 * 2 bits: number of pairs
		 * 1 bit: if has 3 of a kind
		 * 1 bit: if has 4 of a kind
		 * 1 bit: if row
		 * 2 bits: number of undealt cards of primary rank
		 * 2 bits: number of undealt cards of secondsary rank (0 if 3 or more different ranks)
		 * 2 bits: whether have not(0)/exactly(1)/more than(2) enough cards to become a straight
		 * 2 bits: whether have not(0)/exactly(1)/more than(2) enough cards to become a flush
		 */

		// Compute counts
		int numCards = 0;
		int suit = -1; // Suit of first card
		int suitCount = 0; // Number of cards of same suit (0 if there is more than 1 suit)
		int[] rankCounts = new int[Card.NUM_RANKS];
		for (Card card : hand) {
			if (card != null) {
				numCards++;
				rankCounts[card.getRank()]++;
				// Suit count
				if (suit == -1) suit = card.getSuit();
				if (suit == card.getSuit()) suitCount++;
				else {
					// There is more than 1 suit
					suitCount = 0;
					suit = -2; // ignore the suit of other cards
				}
			}
		}
		
		// Compute count of rank counts
		int primaryRankCount = 0;
		int primaryRank = -1;
		int numRanks = 0;
		int[] rankCountCounts = new int[SIZE + 1];
		for (int i=0; i<rankCounts.length; i++) {
			int count = rankCounts[i];
			rankCountCounts[count]++;
			if (count > 0) numRanks++;
			if (count > primaryRankCount) {
				primaryRankCount = count;
				primaryRank = i;
			}
		}

		// Compute secondary rank (if exactly 2 different ranks)
		int secondaryRank = -1;
		if (numRanks == 2) {
			for (int i=0; i<rankCounts.length; i++) {
				if (rankCounts[i]!=0 && i != primaryRank) {
					secondaryRank = i;
					break;
				}
			}
		}

		// Compute rank parts of the abstraction
		int numCardsWithoutPairs = rankCountCounts[1];
		int numPairs = rankCountCounts[2];
		boolean hasThreeOfAKind = rankCountCounts[3] > 0;
		boolean hasFourOfAKind = rankCountCounts[4] > 0;

		// Initial flush/straight checks
		boolean flushPossible = suitCount > 0;
		boolean straightPossible = primaryRankCount <= 1;

		// Straight check
		int[] straightMissingRanks = new int[9];
		int straightNumMissingRanks = 0;
		if (straightPossible) {
			straightPossible = false;
			// Get the smallest rank in the hand
			int minRank = 0;
			while (rankCounts[minRank] == 0) minRank++;

			// Get the largest rank in the hand
			int maxRank = Card.NUM_RANKS - 1;
			while (rankCounts[maxRank] == 0) maxRank--;

			// Straight possible if the space between min and max rank cards is <= 4
			int diff = maxRank - minRank;
			if (diff <= 4) {
				straightPossible = true;

				// Cards needed are the ones missing between min and max AND the surplus before min or after max
				int surplus = 4-diff;
				int startRank = Math.max(minRank-surplus,0);
				int endRank = Math.min(maxRank+surplus,Card.NUM_RANKS-1);
				for (int rank=startRank; rank<=endRank; rank++) {
					if (rankCounts[rank] == 0) {
						straightMissingRanks[straightNumMissingRanks++] = rank;
					}
				}
			}
		}
		
		// Calculate number of important undealt cards using simDeck
		int undealtPrimary = 0;
		int undealtSecondary = 0;
		int undealtFlushCount = 0;
		int[] undealtStraightRanks = new int[straightNumMissingRanks];
		for (int i=numPlays; i<simDeck.length; i++) {
			Card card = simDeck[i];
			if (card.getRank() == primaryRank) undealtPrimary++;
			if (card.getRank() == secondaryRank) undealtSecondary++;
			if (card.getSuit() == suit) undealtFlushCount++;
			for (int j=0; j<straightNumMissingRanks; j++) {
				if (card.getRank() == straightMissingRanks[j]) {
					undealtStraightRanks[j]++;
					break;
				}
			}
		}

		// Simplify undealt straight/flush counters to 0=not enough, 1=exactly enough, 2=more than enough
		int undealtStraight = 0;
		if (straightPossible) {
			// Find the maximum subsequence of available undealt cards that will allow a straight, given sequence length is the number empty spots in the hand
			int seqSize = 5-numCards; // sequence size is number of empty spaces in hand
			int bestSequenceSum = 0;
			int currentSequenceSum = 0;
			int currentSequenceLength = 0;
			for (int i=0; i<undealtStraightRanks.length; i++) {
				if (undealtStraightRanks[i] == 0) {
					currentSequenceSum = 0;
					currentSequenceLength = 0;
				}
				else {
					currentSequenceSum += undealtStraightRanks[i];
					if (currentSequenceLength < seqSize) currentSequenceLength++;
					else currentSequenceSum -= undealtStraightRanks[i-seqSize];
					if (currentSequenceLength == seqSize && currentSequenceSum > bestSequenceSum) bestSequenceSum = currentSequenceSum;
				}
			}
			if (bestSequenceSum>0) undealtStraight = 1;
			if (bestSequenceSum>seqSize) undealtStraight = 2;
		}
		int undealtFlush = 0;
		if (flushPossible) {
			if (undealtFlushCount > SIZE-suitCount) undealtFlush = 2;
			else if (undealtFlushCount == SIZE-suitCount) undealtFlush = 1;
		}

		// Build 16-bit abstraction
		int abstraction = 0;
		abstraction = (abstraction << 3) | numCardsWithoutPairs;
		abstraction = (abstraction << 2) | numPairs;
		abstraction = (abstraction << 1) | (hasThreeOfAKind?1:0);
		abstraction = (abstraction << 1) | (hasFourOfAKind?1:0);
		abstraction = (abstraction << 1) | (isRow?1:0);
		abstraction = (abstraction << 2) | undealtPrimary;
		abstraction = (abstraction << 2) | undealtSecondary;
		abstraction = (abstraction << 2) | undealtStraight;
		abstraction = (abstraction << 2) | undealtFlush;
		return abstraction;
	}

	// Used for debugging
	private String abstractionToString(int abstraction) {
		String result = "";
		result = (abstraction & 0b011) + " " + result; abstraction = abstraction >> 2;
		result = (abstraction & 0b011) + " " + result; abstraction = abstraction >> 2;
		result = (abstraction & 0b011) + " " + result; abstraction = abstraction >> 2;
		result = (abstraction & 0b011) + " " + result; abstraction = abstraction >> 2;
		result = (abstraction & 0b001) + " " + result; abstraction = abstraction >> 1;
		result = (abstraction & 0b001) + " " + result; abstraction = abstraction >> 1;
		result = (abstraction & 0b001) + " " + result; abstraction = abstraction >> 1;
		result = (abstraction & 0b011) + " " + result; abstraction = abstraction >> 2;
		result = (abstraction & 0b111) + " " + result;
		return result;
	}

	// public void testHandAbstraction() {
	// 	boolean isAuto = true;
	// 	boolean hasFailure = false;

	// 	// Manually create test hands
	// 	System.out.println("=== MANUAL TESTS ===");
	// 	int[][][][] testHands = {
	// 		// Tests for: Num pairs & primary/secondary rank
	// 		{{{0,0},{0,1},{0,2},{2,3}}, {{0b0010010101110000}}},
	// 		{{{0,0},{0,1},{2,2},{2,3}}, {{0b0001000010100000}}},
	// 		{{{0,0},{2,1},{2,2},{2,3}}, {{0b0010010101110000}}},
	// 		{{{0,0},{1,1},{2,2},{2,3}}, {{0b0100100010000000}}},
	// 		{{{0,0},{2,1},{1,2},{2,3}}, {{0b0100100110000000}}},
	// 		{{{0,0},{2,1},{2,2},{1,3}}, {{0b0100100010000000}}},

	// 		// Tests for: 3/4 of a kind
	// 		{{{0,0},{0,1},{0,2},{0,3}}, {{0b0000001100000000}}},
	// 		{{{1,0},{0,1},{0,2},{0,3}}, {{0b0010010001110000}}},
	// 		{{{1,0},{0,1},{1,2},{1,3}}, {{0b0010010101110000}}},
	// 		{{{1,0},{1,2},{1,3}}, {{0b0000010001000000}}},

	// 		// Tests for: undealt straight
	// 		{{{0,0}}, {{0b0010000111001010}}},
	// 		{{{0,0},{1,1}}, {{0b0100000011111000}}},
	// 		{{{0,0},{1,1},{2,2}}, {{0b0110000111001000}}},
	// 		{{{0,0},{1,1},{2,2},{3,3}}, {{0b1000000011001000}}},
	// 		{{{6,0},{1,1},{2,2},{3,3}}, {{0b1000000111000000}}},
	// 		{{{6,0},{2,2},{3,3}}, {{0b0110000011001000}}},
	// 		{{{6,0},{9,1}}, {{0b0100000111111000}}},
	// 		{{{9,1}}, {{0b0010000011001010}}},
	// 		{{{8,1}}, {{0b0010000111001010}}},
	// 		{{{0,0},{1,1},{2,2},{3,3},{-1},{4,0},{4,1}}, {{0b1000000011001000}}},
	// 		{{{0,0},{1,1},{2,2},{3,3},{-1},{4,0},{4,1},{4,2}}, {{0b1000000111000100}}},
	// 		{{{0,0},{1,1},{2,2},{3,3},{-1},{4,0},{4,1},{4,2},{4,3}}, {{0b1000000011000000}}},
	// 		{{{6,0},{2,2},{3,3},{-1},{4,0},{4,1},{4,2},{5,0},{5,1},{5,2}}, {{0b0110000111000100}}},
	// 		{{{6,0},{9,1},{-1},{5,0},{5,1},{5,2},{7,0},{7,1},{7,2},{8,0},{8,1},{8,2},{10,0},{10,1},{10,2}}, {{0b0100000011110100}}},
	// 		{{{6,0},{9,1},{-1},{5,0},{5,1},{5,2},{7,0},{7,1},{7,2},{8,0},{8,1},{8,2},{10,0},{10,1}}, {{0b0100000111111000}}},

	// 		// Tests for: undealt flush
	// 		{{{0,0}}, {{0b0010000011001010}}},
	// 		{{{0,0},{1,0}}, {{0b0100000111111010}}},
	// 		{{{0,0},{1,0},{2,0}}, {{0b0110000011001010}}},
	// 		{{{0,0},{1,0},{2,0},{3,0}}, {{0b1000000111001010}}},
	// 		{{{0,0},{1,1}}, {{0b0100000011111000}}},
	// 		{{{0,1},{1,0},{2,0}}, {{0b0110000111001000}}},
	// 		{{{0,0},{1,0},{2,1},{3,0}}, {{0b1000000011001000}}},
	// 		{{{0,0},{1,0},{2,0},{3,0},{-1},{4,0},{5,0},{6,0},{7,0},{8,0},{9,0},{10,0},{11,0},{12,0}}, {{0b1000000111001000}}},
	// 		{{{0,0},{1,0},{2,0},{3,0},{-1},{4,0},{5,0},{6,0},{7,0},{8,0},{9,0},{10,0},{11,0}}, {{0b1000000011001001}}},

	// 	};
	// 	for (int i = 0; i < testHands.length; i++) {
	// 		boolean isRow = (i%2)==0;
	// 		Card[] hand = new Card[5];
	// 		Card[] removed = new Card[52];
	// 		int numRemoved = 0;
	// 		boolean handEnd = false;
	// 		for (int j = 0; j<testHands[i][0].length; j++) {
	// 			int[] testHand = testHands[i][0][j];
	// 			if (testHand[0]==-1) {
	// 				handEnd = true;
	// 				continue;
	// 			}
	// 			Card card = new Card(testHand[0], testHand[1]);
	// 			makePlay(card, 0, 0);
	// 			if (!handEnd) hand[j] = card;
	// 			else removed[numRemoved++] = card;
	// 		}
	// 		// Test
	// 		if (isAuto) {
	// 			int expected = testHands[i][1][0][0];
	// 			int result = getHandAbstraction(hand, isRow);
	// 			if (result != expected) {
	// 				System.out.println("Test case failed: ["+i+"]");
	// 				hasFailure = true;
	// 			}
	// 		}
	// 		else {
	// 			System.out.print("Hand: ");
	// 			for (Card card : hand) {
	// 				if (card!=null) System.out.print(card+" ");
	// 			}
	// 			if (numRemoved > 0) {
	// 				System.out.print("minus [");
	// 				for (int j=0; j<numRemoved; j++) {
	// 					System.out.print(removed[j]+" ");
	// 				}
	// 				System.out.print("]");
	// 			}
	// 			System.out.println();
	// 			int abstraction = getHandAbstraction(hand, isRow);
	// 			// Print abstraction
	// 			String abstractionStr = abstractionToString(abstraction);
	// 			String abstractionBinStr = Integer.toBinaryString(abstraction);
	// 			String abstractionBinStrPad = ("0000000000000000" + abstractionBinStr).substring(abstractionBinStr.length());
	// 			System.out.println("Abstraction: "+abstractionStr+" ("+abstractionBinStrPad+")");
	// 		}
	// 		// Reset after each hand
	// 		init();
	// 	}

	// 	if (isAuto && !hasFailure) {
	// 		System.out.println("All manual tests passed.");
	// 	}

	// 	System.out.print("...");
	// 	Scanner sc = new Scanner(System.in);
	// 	sc.nextLine();
	// 	sc.close();

	// 	System.out.println("=== RANDOM TESTS ===");
	// 	// Generate 10 random hands from 1 deck
	// 	for (int i=0; i<5; i++) {
	// 		Card[] hand = new Card[5];
	// 		int numCards = random.nextInt(4)+1;
	// 		for (int j=0; j<4; j++) {
	// 			if (j>=numCards) {
	// 				hand[j] = null;
	// 			}
	// 			else {
	// 				int cardIndex = random.nextInt(NUM_CARDS - numPlays) + numPlays;
	// 				hand[j] = simDeck[cardIndex];
	// 				makePlay(hand[j], 0, 0);
	// 			}
	// 		}
	// 		System.out.print("Hand: ");
	// 		for (Card card : hand) {
	// 			if (card!=null) System.out.print(card+" ");
	// 		}
	// 		System.out.println();
	// 		boolean isRow = (i%2)==0;
	// 		int abstraction = getHandAbstraction(hand, isRow);
	// 		// Print abstraction
	// 		String abstractionStr = abstractionToString(abstraction);
	// 		String abstractionBinStr = Integer.toBinaryString(abstraction);
	// 		String abstractionBinStrPad = ("0000000000000000" + abstractionBinStr).substring(abstractionBinStr.length());
	// 		System.out.println("Abstraction: "+abstractionStr+" ("+abstractionBinStrPad+")");
	// 	}
	// }

	public static void train(int depth, int iterations) {
		PokerSquaresPointSystem system = PokerSquaresPointSystem.getBritishPointSystem();
		System.out.println(system);
		HARGPMCPlayer player = new HARGPMCPlayer(depth);
		player.isTraining = true;
		PokerSquares ps = new PokerSquares(player, system);
		// ps.setVerbose(false);

		// delta is the exponential decay factor for epsilon (calculated so epsilon reaches 0.1 after all iterations)
		double delta = Math.exp(Math.log(0.1f/player.epsilon)/iterations);
		if (delta > 1) throw new RuntimeException("ERROR: delta>1 ("+delta+")");

		for (int i=0; i<iterations; i++) {
			int score = ps.play();
			System.out.println(score);
			player.epsilon *= delta; // decay epsilon after each iteration
		}
		System.out.println("Done! Completed "+iterations+" iterations. (final epsilon: "+player.epsilon+")");
	}

	public static void main(String[] args) {
		// === Test hand abstrations ===
		// HARGPMCPlayer player = new HARGPMCPlayer(25);
		// player.isTraining = true;
		// player.init();
		// player.testHandAbstraction();

		// Play a single game
		PokerSquaresPointSystem system = PokerSquaresPointSystem.getBritishPointSystem();
		System.out.println(system);
		new PokerSquares(new HARGPMCPlayer(25), system).play(); // play a single game
	}

}
