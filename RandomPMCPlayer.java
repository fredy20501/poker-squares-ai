import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * RandomMCPlayer - a simple Monte Carlo implementation of the player interface for PokerSquares.
 * For each possible play, continues play with random possible card draws and random card placements to a given depth limit 
 * (or game end).  Having sampled trajectories for all possible plays, the RandomMCPlayer then selects the
 * play yielding the best average scoring potential in such Monte Carlo simulation.
 * 
 * Disclaimer: This example code is not intended as a model of efficiency. (E.g., patterns from Knuth's Dancing Links
 * algorithm (DLX) can provide faster legal move list iteration/deletion/restoration.)  Rather, this example
 * code illustrates how a player could be constructed.  Note how time is simply managed so as to not run out the play clock.
 * 
 * Author: Todd W. Neller
 * Modifications by: Michael W. Fleming
 */
public class RandomPMCPlayer implements PokerSquaresPlayer {
	
	private final int SIZE = 5; // number of rows/columns in square grid
	private final int NUM_POS = SIZE * SIZE; // number of positions in square grid
	private final int NUM_CARDS = Card.NUM_CARDS; // number of cards in deck
	private Random random = new Random(); // pseudorandom number generator for Monte Carlo simulation 
	private int[] plays = new int[NUM_POS]; // positions of plays so far (index 0 through numPlays - 1) recorded as integers using row-major indices.
	// row-major indices: play (r, c) is recorded as a single integer r * SIZE + c (See http://en.wikipedia.org/wiki/Row-major_order)
	// From plays index [numPlays] onward, we maintain a list of yet unplayed positions.
	private int numPlays = 0; // number of Cards played into the grid so far
	private PokerSquaresPointSystem system; // point system
	private int depthLimit = 2; // default depth limit for Random Monte Carlo (MC) play
	private Card[][] grid = new Card[SIZE][SIZE]; // grid with Card objects or null (for empty positions)
	private Card[] simDeck = Card.getAllCards(); // a list of all Cards. As we learn the index of cards in the play deck,
	                                             // we swap each dealt card to its correct index.  Thus, from index numPlays 
												 // onward, we maintain a list of undealt cards for MC simulation.
	private int[][] legalPlayLists = new int[NUM_POS][NUM_POS]; // stores legal play lists indexed by numPlays (depth)
	// (This avoids constant allocation/deallocation of such lists during the selections of MC simulations.)

	/**
	 * Create a Random Monte Carlo player that simulates random play to depth 2.
	 */
	public RandomPMCPlayer() {
	}
	
	/**
	 * Create a Random Monte Carlo player that simulates random play to a given depth limit.
	 * @param depthLimit depth limit for random simulated play
	 */
	public RandomPMCPlayer(int depthLimit) {
		this.depthLimit = depthLimit;
	}
	
	/* (non-Javadoc)
	 * @see PokerSquaresPlayer#init()
	 */
	@Override
	public void init() { 
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
			double maxAverageScore = Double.NEGATIVE_INFINITY; // maximum average score found for moves so far
			ArrayList<Integer> bestPlays = new ArrayList<Integer>(); // all plays yielding the maximum average score 
			for (int i = 0; i < remainingPlays; i++) { // for each legal play position
				int play = legalPlayLists[numPlays][i];
				long startTime = System.currentTimeMillis();
				long endTime = startTime + millisPerMoveEval; // compute when MC simulations should end
				makePlay(card, play / SIZE, play % SIZE);  // play the card at the empty position
				int simCount = 0;
				int scoreTotal = 0;
				while (System.currentTimeMillis() < endTime) { // perform as many MC simulations as possible through the allotted time
					// Perform a Monte Carlo simulation of random play to the depth limit or game end, whichever comes first.
					scoreTotal += simPlay(depthLimit);  // accumulate MC simulation scores
					simCount++; // increment count of MC simulations
				}
				undoPlay(); // undo the play under evaluation
				// update (if necessary) the maximum average score and the list of best plays
				double averageScore = (double) scoreTotal / simCount;
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

		int[] playPos = {plays[numPlays] / SIZE, plays[numPlays] % SIZE}; // decode it into row and column
		makePlay(card, playPos[0], playPos[1]); // make the chosen play (not undoing this time)
		return playPos; // return the chosen play
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
			float score = Float.MIN_VALUE;
			float maxScore = Float.MIN_VALUE;
			int depth = Math.min(depthLimit, NUM_POS - numPlays); // compute real depth limit, taking into account game end
			for (int d = 0; d < depth; d++) {
				// generate a random card draw
				int c = random.nextInt(NUM_CARDS - numPlays) + numPlays;
				Card card = simDeck[c];
				// choose a random play from the legal plays

				int remainingPlays = NUM_POS - numPlays;
				System.arraycopy(plays, numPlays, legalPlayLists[numPlays], 0, remainingPlays);
				int c2 = random.nextInt(remainingPlays);
				int play = legalPlayLists[numPlays][c2];
				makePlay(card, play / SIZE, play % SIZE);
			}
			score = getGridUtility();

			// Undo MC plays.
			for (int d = 0; d < depth; d++) {
				undoPlay();
			}

			return score;
		}
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
		else utility = system.getHandScore(hand);
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
			}
			else {
				probArray[5] = probOfRank(rankArrayList.indexOf(2)); //threeOfKind
				probArray[7] = probOfRank(rankArrayList.indexOf(1), rankArrayList.lastIndexOf(1)); // twoPair
			}
		}
		else {
			probArray[3] = probOfSequence(hand, -1); // straight
			probArray[8] = probOfRank(rankArrayList); // pair
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
			return undealtRoyalCardCount/(NUM_CARDS-numPlays);
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
		// 2 -6, diff is 4, now check each card in middle 3,4,5. check each do I have in the hand, one is missing
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
		return eligibleCards/(NUM_CARDS-numPlays);
	}

	float probOfSuit(int suitIndex) {
		int undealtSuitCount = 0;
		for (int i=numPlays; i<simDeck.length; i++) {
			Card card = simDeck[i];
			if (card.getSuit() == suitIndex) undealtSuitCount++;
		}
		return (undealtSuitCount/(NUM_CARDS-numPlays));
	}

	float probOfRank(int rankIndex) {
		int undealtRankCount = 0;
		for (int i=numPlays; i<simDeck.length; i++) {
			Card card = simDeck[i];
			if (card.getRank() == rankIndex) undealtRankCount++;
		}
		return (undealtRankCount/(NUM_CARDS-numPlays));
	}

	float probOfRank(int rankIndex1, int rankIndex2) {
		int undealtRankCount = 0;
		for (int i=numPlays; i<simDeck.length; i++) {
			Card card = simDeck[i];
			if (card.getRank() == rankIndex1) undealtRankCount++;
			else if(card.getRank() == rankIndex2) undealtRankCount++;
		}
		return (undealtRankCount/(NUM_CARDS-numPlays));
	}

	float probOfRank(List<Integer> rankArrayList) {
		int undealtRankCount = 0;
		for (int i=numPlays; i<simDeck.length; i++) {
			Card card = simDeck[i];
			if (rankArrayList.contains(card.getRank())) undealtRankCount++;
		}
		return (undealtRankCount/(NUM_CARDS-numPlays));
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
		return "RandomPMCPlayerDepth" + depthLimit;
	}

	/**
	 * Demonstrate RandomMCPlay with Ameritish point system.
	 * @param args (not used)
	 */
	public static void main(String[] args) {
		PokerSquaresPointSystem system = PokerSquaresPointSystem.getAmeritishPointSystem();
		System.out.println(system);
		new PokerSquares(new RandomPMCPlayer(2), system).play(); // play a single game
	}

}
