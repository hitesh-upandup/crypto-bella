const express = require('express');
const bodyParser = require('body-parser');
const TelegramBot = require('node-telegram-bot-api')
const Levenshtein = require('levenshtein');
const { ChatOpenAI, OpenAIEmbeddings } = require('@langchain/openai');
const { loadQAChain } = require('langchain/chains');
const { PineconeStore } = require('@langchain/pinecone');
const { Pinecone } = require("@pinecone-database/pinecone")
const { Document } = require('langchain/document')

const INDEX_NAME = "test1";
const THRESHOLD = 2;

// process.env.OPENAI_API_KEY = openaiKey;
// process.env.PINECONE_API_KEY = pineconeKey;

const llm = new ChatOpenAI({ temperature: 0, openAIApiKey: process.env.OPENAI_API_KEY, model: "gpt-4" });
const chain = loadQAChain(llm, { type: 'stuff' });

const pineconeClient = new Pinecone({ apiKey: process.env.PINECONE_API_KEY })
pineconeClient.listIndexes().then(list => {
    console.log(list)
})
const index = pineconeClient.Index(INDEX_NAME)
index.describeIndexStats().then(stats => {
    console.log(stats)
})

const bot = new TelegramBot(process.env.TELEGRAM_KEY, {polling: false});

const app = express();

app.use(bodyParser.json());

app.get('/', (req, res) => {
    res.send('I am running fine, how about you');
});

app.post('/webhook', async (req, res) => {
    try {
        const message = req.body;
        const chatId = message.message.chat.id;
        const userInput = message.message.text;

        if (userInput === "/start") {
            bot.sendMessage(chatId, "Ask me questions on cryptocurrency like \"What was the price of Bitcoin on 10 Jan 2024\"?");
        } else {
            const cryptoNames = hasCryptoNames(userInput);
            if (cryptoNames.length > 0) {
                const vectorStore = await PineconeStore.fromExistingIndex(
                    new OpenAIEmbeddings(),
                    { pineconeIndex: index }
                );
                const results = await vectorStore.similaritySearch(userInput, 5)
                reply = await chain.invoke({ question: userInput, input_documents: results })
                console.log(userInput, '\n', reply.text)
                bot.sendMessage(chatId, reply.text)
            } else {
                bot.sendMessage(chatId, "Sorry, I don't know. Please ask me about cryptocurrency only.");
            }
        }
    } catch (err) {
        console.log(err)
    } finally {
        res.send('ok');
    }
});

function hasCryptoNames(inputString) {
    const cryptocurrencyNames = [
        "Cryptocurrency",
        "Bitcoin",
        "Ethereum",
        "Ripple",
        "Litecoin",
        "Cardano",
        "Polkadot",
        "Bitcoin Cash",
        "Chainlink",
        "Stellar",
        "Binance Coin",
        "Monero",
        "Tron",
        "Dash",
        "Dogecoin",
        "EOS",
        "Tezos",
        "NEO",
        "VeChain",
        "Uniswap",
        "Cosmos"
    ];
    const inputStringLower = inputString.toLowerCase();

    const similarNames = [];

    for (const crypto of cryptocurrencyNames) {
        for (const word of inputStringLower.split(' ')) {
            if (new Levenshtein(crypto.toLowerCase(), word).distance <= THRESHOLD) {
                similarNames.push(crypto);
                break;
            }
        }
    }

    return similarNames;
}

app.listen(process.env.PORT, () => {
    console.log(`Server is running on port ${process.env.PORT}`);
});

