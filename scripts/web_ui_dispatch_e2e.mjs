#!/usr/bin/env node
import { createRequire } from "node:module";

const require = createRequire(new URL("../frontend/package.json", import.meta.url));
const { chromium } = require("playwright-core");

const options = parseArgs(process.argv.slice(2));
const browser = await chromium.launch({
  channel: options.channel,
  headless: options.headless,
});

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  page.setDefaultTimeout(options.timeoutMs);
  const websocketUrls = [];
  page.on("websocket", (websocket) => websocketUrls.push(websocket.url()));
  await page.goto(options.baseUrl, { waitUntil: "domcontentloaded" });

  await expectVisible(page, "[data-testid='page-chat']", "Chat page did not render");
  await expectNotVisibleText(page, "[data-testid='page-chat']", "实验工作台", "Chat page leaked workbench copy");

  await sendChatMessage(page, "你是谁");
  await expectVisibleText(page, "[data-testid='chat-transcript']", "AgenticSciML 助手", "Ask identity reply missing");
  await expectNotVisibleText(page, "[data-testid='chat-transcript']", "actions", "Ask mode returned visible actions");

  await page.locator("[data-testid='assistant-mode-plan']").first().click();
  await sendChatMessage(page, "打开当前账号代码工作区");
  await expectVisibleText(page, "[data-testid='chat-transcript']", "open_code_server", "Plan mode did not expose proposed code action");
  await expectVisible(page, "[data-testid='page-chat']", "Plan mode should not navigate away from Chat page");

  await page.locator("[data-testid='assistant-mode-agent']").first().click();
  const codeServerWebsocketPromise = options.expectCodeServerWebsocket
    ? waitForCodeServerWebsocket(page, options.websocketTimeoutMs)
    : null;
  await sendChatMessage(page, "打开当前账号代码工作区");
  await expectVisible(page, "[data-testid='page-ide']", "Agent code action did not navigate to IDE page");
  await expectVisible(page, "[data-testid='vscode-iframe']", "IDE page did not open the VS Code iframe");
  const codeServerWebsocketUrl = codeServerWebsocketPromise ? await codeServerWebsocketPromise : null;
  await expectNotVisibleText(page, "[data-testid='page-ide']", "实验工作台", "IDE page leaked workbench copy");

  await page.locator("[data-testid='page-ide'] [data-testid='assistant-mode-ask']").click();
  await page.locator("[data-testid='ide-agent-input']").fill("你能做什么");
  await page.locator("[data-testid='ide-agent-send']").click();
  await expectVisibleText(page, "[data-testid='page-ide']", "Plan 模式", "IDE sidebar ChatUI did not answer in Ask mode");

  await page.locator("[data-testid='nav-library']").click();
  await expectVisible(page, "[data-testid='page-library']", "Library page did not render");
  await expectVisibleText(page, "[data-testid='page-library']", "实验工作台", "Library page should contain workbench copy");

  console.log(
    JSON.stringify(
      {
        ok: true,
        base_url: options.baseUrl,
        checks: [
          "chat_page_copy_boundary",
          "ask_no_actions",
          "plan_no_dispatch",
          "agent_open_code_server_dispatch",
          "ide_sidebar_chatui_ask",
          "library_workbench_boundary",
        ],
        code_server_websocket_url: codeServerWebsocketUrl,
        observed_websocket_count: websocketUrls.length,
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}

async function sendChatMessage(page, text) {
  await page.locator("[data-testid='chat-main-input']").fill(text);
  await page.locator("[data-testid='chat-main-send']").click();
}

async function expectVisible(page, selector, message) {
  await page.locator(selector).waitFor({ state: "visible" }).catch((error) => {
    throw new Error(`${message}: ${error.message}`);
  });
}

async function expectVisibleText(page, selector, text, message) {
  await page.locator(selector).getByText(text, { exact: false }).first().waitFor({ state: "visible" }).catch((error) => {
    throw new Error(`${message}: ${error.message}`);
  });
}

async function expectNotVisibleText(page, selector, text, message) {
  const count = await page.locator(selector).getByText(text, { exact: false }).count();
  if (count !== 0) {
    throw new Error(`${message}: found ${count} visible match(es) for ${JSON.stringify(text)}`);
  }
}

async function waitForCodeServerWebsocket(page, timeoutMs) {
  const websocket = await page.waitForEvent("websocket", {
    timeout: timeoutMs,
    predicate: (candidate) => {
      const url = candidate.url();
      return url.startsWith("ws://") || url.startsWith("wss://");
    },
  }).catch((error) => {
    throw new Error(`code-server WebSocket did not open through the page: ${error.message}`);
  });
  return websocket.url();
}

function parseArgs(args) {
  const parsed = {
    baseUrl: "http://127.0.0.1:5173",
    channel: "chrome",
    headless: true,
    timeoutMs: 30000,
    websocketTimeoutMs: 60000,
    expectCodeServerWebsocket: false,
  };
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "--base-url") {
      parsed.baseUrl = requireValue(args, ++i, arg).replace(/\/$/, "");
    } else if (arg === "--channel") {
      parsed.channel = requireValue(args, ++i, arg);
    } else if (arg === "--timeout-ms") {
      parsed.timeoutMs = Number(requireValue(args, ++i, arg));
    } else if (arg === "--websocket-timeout-ms") {
      parsed.websocketTimeoutMs = Number(requireValue(args, ++i, arg));
    } else if (arg === "--expect-code-server-websocket") {
      parsed.expectCodeServerWebsocket = true;
    } else if (arg === "--headed") {
      parsed.headless = false;
    } else if (arg === "--headless") {
      parsed.headless = true;
    } else if (arg === "--help" || arg === "-h") {
      console.log("Usage: node scripts/web_ui_dispatch_e2e.mjs [--base-url URL] [--channel chrome] [--expect-code-server-websocket] [--headed]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function requireValue(args, index, option) {
  const value = args[index];
  if (!value) throw new Error(`${option} requires a value`);
  return value;
}
