This is a project for training a transformer model on chess games in an autoregressive fashion.

<project_info>
The are multiple components (more to come):
# ./data-processing/
In here is all the preprocessing code for creating the training data from the games

# ./games
Here is where the raw unprocessed games data is stored. It consists of pgn file per month from lichess. The .zstd files should never be read by a script, only the decompressed games should be taken as input

# ./logs
Here is where we keep logs from training runs so that we can track progress. Progress should also be tracked to wandb, but text logs should be written here so that you we can follow up on the experiments and track what is working vs not.

# ./todo.md
Here we track all the tasks that needs to be done. Feel free to write to this when you find issues that needs to be solved in the future so that we can tackle them later.

</project_info>

<guidelines>
 - Dont use exessive commenting on all the code. Only comment the code if its hard to understand or it does things that are unintive.
 - Use the AskUserQuestion Tool anytime there is any ambuigity in the user message. Continue to ask questions until the plan is very clear
 - Aim for writing tight, easy to read code. It should be simple in its nature. Try to keep functions short, splitting when it makes sense. Avoid exessive abstractions, prefer simple code. Avoid using uneccesary classes, prefer simpler functional programming.
 - After making changes always commit them so that we can restore progress if we need to.
 - Aim for having full test coverage where it makes sense.
 - Always run the lint and make sure the code passes.
</guidelines>
