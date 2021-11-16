import java.io.RandomAccessFile;
import java.io.IOException;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.PrintWriter;

public class TestRunner {

    // Change these values to test a different player & log the results in a different file
    private static final File logDir = new File("logs");
    private static final File logFile = new File(logDir.getName()+"/RandomPlayer.log");
    private static final PokerSquaresPlayer player = new RandomPlayer();

    public static void main(String[] args) {
        // Poker square setup
        PokerSquaresPointSystem.setSeed(0L);
        PokerSquaresPointSystem system = PokerSquaresPointSystem.getBritishPointSystem();
        PokerSquares ps = new PokerSquares(player, system);
        ps.setVerbose(false);

        initLogFile();

        // Run simulations forever (until stopped manually)
        while (true) {
            int score = ps.play();
            saveScore(score);
        }
    }

    // If log file doesn't exist, create it with initial values.
    private static void initLogFile() {
        // Create logs directory if doesn't exist
        if (!logDir.exists()) logDir.mkdir();
        // Check if file exists
        if (!logFile.isFile()) {
            // Create file with initial values
            String header = "Play# Score Min Max Cummulative_Mean Cummulative_StdDev S(n)";
            String firstLine = "0 0 0 0 0 0 0";
            addLine(header);
            addLine(firstLine);
        }
    }

    // Read the current values in log file and calculate next ones using given score
    private static void saveScore(int score) {
        // Get last values from log file
        String[] values = tail(logFile).split(" ");
        int numPlay = Integer.parseInt(values[0]) + 1;
        int prevMin = Integer.parseInt(values[2]);
        int prevMax = Integer.parseInt(values[3]);
        double prevMean = Double.parseDouble(values[4]);
        double prevS = Double.parseDouble(values[6]);

        // Calculate new values (formulas taken from: https://datagenetics.com/blog/november22017/index.html)
        int newMin, newMax;
        double newMean, newS, newStdDev;
        if (numPlay == 0) {
            // Special case for first value
            newMin = score;
            newMax = score;
            newMean = score;
            newS = 0;
            newStdDev = 0;
        }
        else {
            newMin = score < prevMin ? score : prevMin;
            newMax = score > prevMax ? score : prevMax;
            newMean = prevMean + (score - prevMean)/numPlay;
            newS = prevS + (score - prevMean)*(score - newMean);
            newStdDev = Math.sqrt(newS/numPlay);
        }

        // Append new values to log file (and print to console)
        String newLine = numPlay+" "+score+" "+newMin+" "+newMax+" "+newMean+" "+newStdDev+" "+newS;
        addLine(newLine);
    }

    // Append the given line to log file (creates file if doesn't exist)
    // Also print the line to the console
    private static void addLine(String line) {
        System.out.println(line);
        try(FileWriter fw = new FileWriter(logFile, true);
            BufferedWriter bw = new BufferedWriter(fw);
            PrintWriter out = new PrintWriter(bw))
        {
            out.println(line);
        } catch (IOException e) {
            // ¯\_(ツ)_/¯
            System.out.println(e.getMessage());
        }
    }

    // Return the last line of the given file as a string
    // Taken from: https://stackoverflow.com/a/7322581
    private static String tail( File file ) {
        RandomAccessFile fileHandler = null;
        try {
            fileHandler = new RandomAccessFile( file, "r" );
            long fileLength = fileHandler.length() - 1;
            StringBuilder sb = new StringBuilder();
    
            for(long filePointer = fileLength; filePointer != -1; filePointer--){
                fileHandler.seek( filePointer );
                int readByte = fileHandler.readByte();
    
                if( readByte == 0xA ) {
                    if( filePointer == fileLength ) {
                        continue;
                    }
                    break;
                    
                } else if( readByte == 0xD ) {
                    if( filePointer == fileLength - 1 ) {
                        continue;
                    }
                    break;
                }
    
                sb.append( ( char ) readByte );
            }
    
            String lastLine = sb.reverse().toString();
            return lastLine;
        } catch( java.io.FileNotFoundException e ) {
            e.printStackTrace();
            return null;
        } catch( java.io.IOException e ) {
            e.printStackTrace();
            return null;
        } finally {
            if (fileHandler != null ) {
                try {
                    fileHandler.close();
                } catch (IOException e) {
                    /* ignore */
                }
            }
        }
    }
}
