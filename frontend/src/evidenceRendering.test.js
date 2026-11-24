import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { fileURLToPath } from "node:url";

import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

let vite;
let components;

before(async () => {
  vite = await createServer({
    appType: "custom",
    configFile: false,
    root: fileURLToPath(new URL("..", import.meta.url)),
    server: { middlewareMode: true },
  });
  components = await vite.ssrLoadModule("/src/App.jsx");
});

after(async () => {
  if (vite) await vite.close();
});

function render(component, props) {
  return renderToStaticMarkup(React.createElement(component, props));
}

function interviewRoomProps(question, evaluation = null) {
  return {
    session: {
      role: "Backend Engineer",
      interview_type: "technical",
      number_of_questions: 2,
      current_question: 0,
    },
    question,
    answer: "I used a worker pool to reduce queue delay.",
    evaluation,
    loading: false,
    setAnswer() {},
    onSubmit() {},
    onNext() {},
    onReport() {},
    onExit() {},
  };
}

test("interview room displays question evidence and omits it for general interviews", () => {
  const documentQuestion = render(
    components.InterviewRoom,
    interviewRoomProps({
      question: "How did you improve the service?",
      evidence: [
        {
          filename: "resume.pdf",
          page_number: 4,
          text: "Built a queue worker pool to reduce latency.",
        },
      ],
    }),
  );

  assert.match(documentQuestion, /Question source evidence/);
  assert.match(documentQuestion, /resume\.pdf/);
  assert.match(documentQuestion, /Page 4/);
  assert.match(documentQuestion, /Built a queue worker pool to reduce latency\./);

  const generalQuestion = render(
    components.InterviewRoom,
    interviewRoomProps({ question: "How do you approach debugging?" }),
  );
  assert.doesNotMatch(generalQuestion, /evidence-list|Source document/);
});

test("evaluation distinguishes the candidate answer, feedback, and source evidence", () => {
  const markup = render(components.InterviewRoom, interviewRoomProps(
    {
      question: "Explain the queue design.",
      evidence: [
        {
          filename: "notes.pdf",
          page_number: 2,
          text: "A queue processes waiting tasks in FIFO order.",
        },
      ],
    },
    {
      score: 8,
      technical_correctness: 8,
      relevance: 8,
      completeness: 7,
      clarity: 8,
      strengths: ["Clear reasoning"],
      weaknesses: ["Add a concrete example"],
      improvement_suggestions: ["Explain the trade-off."],
      evidence: [
        {
          filename: "notes.pdf",
          page_number: 3,
          text: "Queue consumers process messages in arrival order.",
        },
      ],
      next_question_available: true,
    },
  ));

  assert.match(markup, /Your answer/);
  assert.match(markup, /I used a worker pool to reduce queue delay\./);
  assert.match(markup, /What went well/);
  assert.match(markup, /Question source evidence/);
  assert.match(markup, /Evaluation source evidence/);
  assert.match(markup, /Page 3/);
  assert.match(markup, /Queue consumers process messages in arrival order\./);
});

test("reports show both evidence sources and merge duplicate references", () => {
  const sharedEvidence = {
    document_id: "study-1",
    filename: "notes.pdf",
    page_number: 5,
    chunk_id: "chunk-shared",
    text: "A queue follows FIFO ordering.",
  };
  const markup = render(components.Report, {
    report: {
      overall_score: 8,
      role: "Backend Engineer",
      question_count: 1,
      interview_type: "technical",
      strengths: [],
      weaknesses: [],
      recommendations: [],
      question_results: [
        {
          question_number: 1,
          question: "Explain queue ordering.",
          answer: "A queue processes items in arrival order.",
          score: 8,
          question_evidence: [
            sharedEvidence,
            {
              ...sharedEvidence,
              page_number: 6,
              chunk_id: "chunk-question",
              text: "New question-specific excerpt.",
            },
          ],
          evaluation_evidence: [
            sharedEvidence,
            {
              ...sharedEvidence,
              page_number: 7,
              chunk_id: "chunk-evaluation",
              text: "New evaluation-specific excerpt.",
            },
          ],
        },
      ],
    },
    loading: false,
    onHistory() {},
    onDashboard() {},
  });

  assert.match(markup, /Question and evaluation/);
  assert.match(markup, /New question-specific excerpt\./);
  assert.match(markup, /New evaluation-specific excerpt\./);
  assert.equal(markup.split("A queue follows FIFO ordering.").length - 1, 1);

  const generalReport = render(components.Report, {
    report: {
      overall_score: 7,
      role: "Backend Engineer",
      question_count: 1,
      interview_type: "general",
      strengths: [],
      weaknesses: [],
      recommendations: [],
      question_results: [
        {
          question_number: 1,
          question: "How do you approach debugging?",
          answer: "I isolate the smallest failing case.",
          score: 7,
        },
      ],
    },
    loading: false,
    onHistory() {},
    onDashboard() {},
  });
  assert.doesNotMatch(generalReport, /evidence-list|Source document/);
});

test("evidence handles missing document metadata and excerpts", () => {
  const markup = render(components.InterviewRoom, interviewRoomProps({
    question: "How would you explain this result?",
    evidence: [{}],
  }));

  assert.match(markup, /Source document/);
  assert.match(markup, /Excerpt unavailable\./);
  assert.doesNotMatch(markup, /Page undefined/);
});
