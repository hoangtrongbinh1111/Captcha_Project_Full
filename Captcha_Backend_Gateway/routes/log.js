const express = require("express");
const router = express.Router();

const cleanBody = require("../middlewares/cleanbody");
const { validateToken } = require("../middlewares/validateToken");

const LogController = require("../src/log/log.controller");

router.get("/list", LogController.ListLogs);
router.get("/extract", LogController.readLog);
router.post("/detect", cleanBody, LogController.DetectMaliciousWebsite);

module.exports = router;