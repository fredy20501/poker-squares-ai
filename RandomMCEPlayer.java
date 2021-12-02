import java.util.ArrayList;
import java.util.Random;

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
public class RandomMCEPlayer implements PokerSquaresPlayer {

	private final int SIZE = 5; // number of rows/columns in square grid
	private final int NUM_POS = SIZE * SIZE; // number of positions in square grid
	private final int NUM_CARDS = Card.NUM_CARDS; // number of cards in deck
	private final int EXPECTISIM_DEPTH = 4; // number of moves INCLUDING FINAL MOVE at the end of the game to be calculated by expectimax.
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
	public RandomMCEPlayer() {
	}

	/**
	 * Create a Random Monte Carlo player that simulates random play to a given depth limit.
	 * @param depthLimit depth limit for random simulated play
	 */
	public RandomMCEPlayer(int depthLimit) {
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
		int remainingPlays = NUM_POS - numPlays;

		while (!card.equals(simDeck[cardIndex]))
			cardIndex++;
		simDeck[cardIndex] = simDeck[numPlays];
		simDeck[numPlays] = card;

		// Ben Myles
		// 2021-11-18
		if (remainingPlays == 1) {
			// Do nothing to allow forced play
		}
		else if (remainingPlays <= EXPECTISIM_DEPTH) { // Last few turns will be calculated by expectimax.
			float maxScore = 0;
			System.arraycopy(plays, numPlays, legalPlayLists[numPlays], 0, remainingPlays);
			ArrayList<Integer> bestPlays = new ArrayList<Integer>(); // all plays yielding the maximum average score

			//System.out.println("DEBUG EXPECTIMAX----------------------------");
			for (int i = 0; i < remainingPlays; i++) { // for each legal play position
				// Try a play, simulate to the end of the game
				int play = legalPlayLists[numPlays][i];
				// System.out.printf("\nTrying play: (%d, %d) = %s\n", play/SIZE, play%SIZE, card);
				makePlay(card, play / SIZE, play % SIZE);  // play the card at the empty position
				float expectedScore = expectiSimPlay();
				// System.out.printf("Expected score: %.1f\n", expectedScore);
				// Use the expected score of the simulated play to rank the play
				if (expectedScore >= maxScore) {
					if (expectedScore > maxScore) {
						bestPlays.clear();
					}
					bestPlays.add(play);
					maxScore = expectedScore;
				}
				undoPlay();
			}
			// System.out.println("END DEBUG EXPECTIMAX------------------------");
			int bestPlay = bestPlays.get(random.nextInt(bestPlays.size())); // choose a best play (breaking ties randomly)
			// update our list of plays, recording the chosen play in its sequential position; all onward from numPlays are empty positions
			int bestPlayIndex = numPlays;
			while (plays[bestPlayIndex] != bestPlay) {
				bestPlayIndex++;
			}
			plays[bestPlayIndex] = plays[numPlays];
			plays[numPlays] = bestPlay;
		}
		// End

		else if (remainingPlays > EXPECTISIM_DEPTH) { // not the last few plays.
			// compute average time per move evaluation
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
		// System.out.printf("Making play: (%d, %d) = %s\n", plays[numPlays] / SIZE, plays[numPlays] % SIZE, card);
		makePlay(card, playPos[0], playPos[1]); // make the chosen play (not undoing this time)
		return playPos; // return the chosen play
	}

	/*
	 * From the chosen play, attempt every possible game-end, evaluating them to
	 * determine the expected score of the chosen play
	 * @return resulting grid score after playing until end of game
	*/
	private float expectiSimPlay() {
		// Base case: Expected value of a completed game is the score of the game
		if (numPlays == NUM_POS) {
			return system.getScore(grid);
		}
		int remainingPlays = NUM_POS - numPlays;

		// Loop over the remaining cards in the deck, adding up their expected values
		int sum = 0;
		for (int i=numPlays; i<NUM_CARDS; i++) {
			Card card = simDeck[i];

			// Loop over the legal plays
			System.arraycopy(plays, numPlays, legalPlayLists[numPlays], 0, remainingPlays);
			for (int j=0; j<remainingPlays; j++) {
				int play = legalPlayLists[numPlays][j];
				makePlay(card, play / SIZE, play % SIZE);
				// System.out.printf("%s: %.1f\n", card, expectiSimPlay());
				sum += expectiSimPlay()/remainingPlays; // Returns the expected value of this card being played. Divide by remaining plays to scale the returned value by the number of moves that it's testing
				undoPlay();
			}
		}
		// Calculate expected score by averaging the expected score of all the possible cards that could be pulled
		return ((float)sum)/(NUM_CARDS-numPlays);
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
		return "RandomMCEPlayerDepth" + depthLimit;
	}

	/**
	 * Demonstrate RandomMCPlay with Ameritish point system.
	 * @param args (not used)
	 */
	public static void main(String[] args) {
		PokerSquaresPointSystem system = PokerSquaresPointSystem.getBritishPointSystem();
		System.out.println(system);
		new PokerSquares(new RandomMCEPlayer(25), system).play(); // play a single game
	}

}
