import 'dotenv/config';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { pull } from 'langchain/hub';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';

const gptModel = new ChatOpenAI({
  modelName: 'gpt-4',
  temperature: 0,
});

// Load
const loader = new CheerioWebBaseLoader(
  'https://circleci.com/blog/introduction-to-graphql/',
  {}
);

const documents = await loader.load();

// Split
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1200,
  chunkOverlap: 200,
});

const webPageSplits = await splitter.splitDocuments(documents);

// Embed
const embedder = new OpenAIEmbeddings();

// Store
const store = await MemoryVectorStore.fromDocuments(webPageSplits, embedder);

// Retrieve
const retriever = store.asRetriever();

const prompt = ChatPromptTemplate.fromTemplate(`
Answer the following question based on the provided context. If the answer cannot be found in the context, say "I cannot find this information in the provided context."

Context: {context}

Question: {input}

Answer the question in a detailed way based on the context provided.
`);

const documentChain = await createStuffDocumentsChain({
  llm: gptModel,
  prompt,
});

const retrievalChain = await createRetrievalChain({
  combineDocsChain: documentChain,
  retriever,
});

const response = await retrievalChain.invoke({
  input: 'How is GraphQL compared to REST?',
});

console.log(response.answer);
