This is a project for training a transformer model on chess games in an autoregressive fashion.

<project_info>
The are multiple components (more to come):
# ./data-processing/
In here is all the preprocessing code for creating the training data from the games


# ./games
Here is where the raw unprocessed games data is stored. It consists of pgn file per month from lichess. The .zstd files should never be read by a script, only the decompressed games should be taken as input
</project_info>

<guidelines>
 - Dont use exessive commenting on all the code. Only comment the code if its hard to understand or it does things that are unintive.
 - Use the AskUserQuestion Tool anytime there is any ambuigity in the user message. Continue to ask questions until the plan is very clear
 - Aim for writing tight, easy to read code. It should be simple in its nature. Try to keep functions short, splitting when it makes sense. Avoid exessive abstractions, prefer simple code. Avoid using uneccesary classes, prefer simpler functional programming.
</guidelines>
