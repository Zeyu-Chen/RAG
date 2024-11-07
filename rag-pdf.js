import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import 'dotenv/config';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import 'pdf-parse';

const gptModel = new ChatOpenAI({
  modelName: 'gpt-4',
  temperature: 0,
});

// Load
const loader = new PDFLoader('./test.pdf');

const documents = await loader.load();

// Split
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1200,
  chunkOverlap: 200,
});

const pdfSplits = await splitter.splitDocuments(documents);

// Embed
const embedder = new OpenAIEmbeddings();

// Store
const store = await MemoryVectorStore.fromDocuments(pdfSplits, embedder);

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
  input: 'What is Zeyu previous work?',
});

console.log(response.answer);
