import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

/**
 * (Hand Abstraction Greedy Monte Carlo Player)
 * HAGMCPlayer - a simple  Monte Carlo implementation of the player interface for PokerSquares using an 
 * optimized greedy search as a playout policy and hand abstractions for partial evaluation.
 */
public class HAGMCPlayer implements PokerSquaresPlayer {
	
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
	public HAGMCPlayer() {}
	
	/**
	 * Create a Random Monte Carlo player that simulates random play to a given depth limit.
	 * @param depthLimit depth limit for random simulated play
	 */
	public HAGMCPlayer(int depthLimit) {
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
	private int simPlay(int depthLimit) {
		if (depthLimit == 0) { // with zero depth limit, return current score
			return system.getScore(grid);
		}
		else { // up to the non-zero depth limit or to game end, iteratively make the given number of random plays 
			int score = Integer.MIN_VALUE;
			int maxScore = Integer.MIN_VALUE;
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
			score = system.getScore(grid);

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
		return "HAGMCPlayerDepth" + depthLimit;
	}

	private int getScore(Card[][] grid) {
		return 0;
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
		 * 2 bits: whether have not(0)/exactly(1)/more than(2) enough cards to become a flush
		 * 2 bits: whether have not(0)/exactly(1)/more than(2) enough cards to become a straight
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


		// ! DEBUG CHECK (REMOVE ME)
		// ! DEBUG CHECK (REMOVE ME)
		// ! DEBUG CHECK (REMOVE ME)
		// ! DEBUG CHECK (REMOVE ME)
		if (numCardsWithoutPairs>4) System.out.println("[ERROR] numCardsWithoutPairs invalid: "+numCardsWithoutPairs);
		if (numPairs>2) System.out.println("[ERROR] numPairs invalid: "+numPairs);
		if (undealtPrimary>3) System.out.println("[ERROR] undealtPrimary invalid: "+undealtPrimary);
		if (undealtSecondary>3) System.out.println("[ERROR] undealtSecondary invalid: "+undealtSecondary);
		if (undealtStraight>2) System.out.println("[ERROR] undealtStraight invalid: "+undealtStraight);
		if (undealtFlush>2) System.out.println("[ERROR] undealtFlush invalid: "+undealtFlush);
		System.out.println(
			numCardsWithoutPairs+" "+
			numPairs+" "+
			(hasThreeOfAKind?1:0)+" "+
			(hasFourOfAKind?1:0)+" "+
			(isRow?1:0)+" "+
			undealtPrimary+" "+
			undealtSecondary+" "+
			undealtStraight+" "+
			undealtFlush
		);
		// ! DEBUG CHECK (REMOVE ME)
		// ! DEBUG CHECK (REMOVE ME)
		// ! DEBUG CHECK (REMOVE ME)
		// ! DEBUG CHECK (REMOVE ME)


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

	private int evaluateHandAbstraction(int handAbstraction) {
		return 0;
	}

	public void testHandAbstraction() {

		// Manually create test hands
		System.out.println("=== MANUAL TESTS ===");
		int[][][] testHands = {
			// Tests for: Num pairs & primary/secondary rank
			// {{0,0},{0,1},{0,2},{2,3}},
			// {{0,0},{0,1},{2,2},{2,3}},
			// {{0,0},{2,1},{2,2},{2,3}},
			// {{0,0},{1,1},{2,2},{2,3}},
			// {{0,0},{2,1},{1,2},{2,3}},
			// {{0,0},{2,1},{2,2},{1,3}},

			// Tests for: 3/4 of a kind
			// {{0,0},{0,1},{0,2},{0,3}},
			// {{1,0},{0,1},{0,2},{0,3}},
			// {{1,0},{0,1},{1,2},{1,3}},
			// {{1,0},{1,2},{1,3}},

			// Tests for: undealt straight
			// {{0,0}},
			// {{0,0},{1,1}},
			// {{0,0},{1,1},{2,2}},
			// {{0,0},{1,1},{2,2},{3,3}},
			// {{6,0},{1,1},{2,2},{3,3}},
			// {{6,0},{2,2},{3,3}},
			// {{6,0},{9,1}},
			// {{9,1}},
			// {{8,1}},
			// {{0,0},{1,1},{2,2},{3,3},{-1},{4,0},{4,1}},
			// {{0,0},{1,1},{2,2},{3,3},{-1},{4,0},{4,1},{4,2}},
			// {{0,0},{1,1},{2,2},{3,3},{-1},{4,0},{4,1},{4,2},{4,3}},
			// {{6,0},{2,2},{3,3},{-1},{4,0},{4,1},{4,2},{5,0},{5,1},{5,2}},
			// {{6,0},{9,1},{-1},{5,0},{5,1},{5,2},{7,0},{7,1},{7,2},{8,0},{8,1},{8,2},{10,0},{10,1},{10,2}},
			// {{6,0},{9,1},{-1},{5,0},{5,1},{5,2},{7,0},{7,1},{7,2},{8,0},{8,1},{8,2},{10,0},{10,1}},

			// Tests for: undealt flush
			// Todo

		};
		for (int i = 0; i < testHands.length; i++) {
			Card[] hand = new Card[5];
			Card[] removed = new Card[52];
			int numRemoved = 0;
			boolean handEnd = false;
			for (int j = 0; j<testHands[i].length; j++) {
				int[] testHand = testHands[i][j];
				if (testHand[0]==-1) {
					handEnd = true;
					continue;
				}
				Card card = new Card(testHand[0], testHand[1]);
				makePlay(card, 0, 0);
				if (!handEnd) hand[j] = card;
				else removed[numRemoved++] = card;
			}
			// Test
			System.out.print("Hand: ");
			for (Card card : hand) {
				if (card!=null) System.out.print(card+" ");
			}
			if (numRemoved > 0) {
				System.out.print("minus [");
				for (int j=0; j<numRemoved; j++) {
					System.out.print(removed[j]+" ");
				}
				System.out.print("]");
			}
			System.out.println();
			boolean isRow = (i%2)==0;
			getHandAbstraction(hand, isRow);
			// Reset after each hand
			init();
		}

		System.out.print("...");
		Scanner sc = new Scanner(System.in);
		sc.nextLine();
		sc.close();

		System.out.println("=== RANDOM TESTS ===");
		// Generate 10 random hands from 1 deck
		for (int i=0; i<5; i++) {
			Card[] hand = new Card[5];
			int numCards = random.nextInt(4)+1;
			for (int j=0; j<4; j++) {
				if (j>=numCards) {
					hand[j] = null;
				}
				else {
					int cardIndex = random.nextInt(NUM_CARDS - numPlays) + numPlays;
					hand[j] = simDeck[cardIndex];
					makePlay(hand[j], 0, 0);
				}
			}
			System.out.print("Hand: ");
			for (Card card : hand) {
				if (card!=null) System.out.print(card+" ");
			}
			System.out.println();
			boolean isRow = (i%2)==0;
			getHandAbstraction(hand, isRow);

			// int abstraction = getHandAbstraction(hand, isRow);
			// String abstractionStr = Integer.toBinaryString(abstraction);
			// System.out.println("Abstraction: "+("0000000000000000" + abstractionStr).substring(abstractionStr.length()));
		}
	}

	/**
	 * Demonstrate RandomMCPlay with Ameritish point system.
	 * @param args (not used)
	 */
	public static void main(String[] args) {
		// PokerSquaresPointSystem system = PokerSquaresPointSystem.getBritishPointSystem();
		// System.out.println(system);
		// new PokerSquares(new HAGMCPlayer(2), system).play(); // play a single game

		// === Test hand abstrations ===
		HAGMCPlayer player = new HAGMCPlayer(2);
		player.init();
		player.testHandAbstraction();
	}

}
