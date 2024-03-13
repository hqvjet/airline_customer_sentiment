"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
self["webpackHotUpdate_N_E"]("app/show-model/[emb]/[model]/page",{

/***/ "(app-pages-browser)/./src/apis/index.ts":
/*!***************************!*\
  !*** ./src/apis/index.ts ***!
  \***************************/
/***/ (function(module, __webpack_exports__, __webpack_require__) {

eval(__webpack_require__.ts("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   GLOVE_CONFIG: function() { return /* binding */ GLOVE_CONFIG; },\n/* harmony export */   PHOBERT_CONFIG: function() { return /* binding */ PHOBERT_CONFIG; },\n/* harmony export */   \"default\": function() { return /* binding */ useAPI; }\n/* harmony export */ });\n/* harmony import */ var axios__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! axios */ \"(app-pages-browser)/./node_modules/axios/lib/axios.js\");\n/* harmony import */ var _glove__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./glove */ \"(app-pages-browser)/./src/apis/glove.ts\");\n/* harmony import */ var _phobert__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./phobert */ \"(app-pages-browser)/./src/apis/phobert.ts\");\n/* provided dependency */ var process = __webpack_require__(/*! process */ \"(app-pages-browser)/./node_modules/next/dist/build/polyfills/process.js\");\n\n\n\nconst GLOVE_CONFIG = axios__WEBPACK_IMPORTED_MODULE_2__[\"default\"].create({\n    baseURL: process.env.SERVER_API + \"/glove\",\n    timeout: 10000\n});\nconst PHOBERT_CONFIG = axios__WEBPACK_IMPORTED_MODULE_2__[\"default\"].create({\n    baseURL: process.env.SERVER_API + \"/phobert\",\n    timeout: 10000\n});\nfunction useAPI(path) {\n    path = path.split(\"\");\n    const emb = path[2];\n    const model = path[3];\n    console.log(pathemb, model);\n    if (emb == \"glove\") {\n        if (model == \"cnn\") return _glove__WEBPACK_IMPORTED_MODULE_0__.usingGLOVE.usingCNN;\n        else if (model == \"lstm\") return _glove__WEBPACK_IMPORTED_MODULE_0__.usingGLOVE.usingLSTM;\n        else if (model == \"bilstm\") return _glove__WEBPACK_IMPORTED_MODULE_0__.usingGLOVE.usingBILSTM;\n        else if (model == \"ensemble_cnn_bilstm\") return _glove__WEBPACK_IMPORTED_MODULE_0__.usingGLOVE.usingENSEMBLE_CNN_BILSTM;\n        else if (model == \"fusion_cnn_bilstm\") return _glove__WEBPACK_IMPORTED_MODULE_0__.usingGLOVE.usingFUSION_CNN_BILSTM;\n    } else if (emb == \"phobert\") {\n        if (model == \"cnn\") return _phobert__WEBPACK_IMPORTED_MODULE_1__.usingPHOBERT.usingCNN;\n        else if (model == \"lstm\") return _phobert__WEBPACK_IMPORTED_MODULE_1__.usingPHOBERT.usingLSTM;\n        else if (model == \"bilstm\") return _phobert__WEBPACK_IMPORTED_MODULE_1__.usingPHOBERT.usingBILSTM;\n        else if (model == \"ensemble_cnn_bilstm\") return _phobert__WEBPACK_IMPORTED_MODULE_1__.usingPHOBERT.usingENSEMBLE_CNN_BILSTM;\n        else if (model == \"fusion_cnn_bilstm\") return _phobert__WEBPACK_IMPORTED_MODULE_1__.usingPHOBERT.usingFUSION_CNN_BILSTM;\n    }\n    return null;\n}\n\n\n;\n    // Wrapped in an IIFE to avoid polluting the global scope\n    ;\n    (function () {\n        var _a, _b;\n        // Legacy CSS implementations will `eval` browser code in a Node.js context\n        // to extract CSS. For backwards compatibility, we need to check we're in a\n        // browser context before continuing.\n        if (typeof self !== 'undefined' &&\n            // AMP / No-JS mode does not inject these helpers:\n            '$RefreshHelpers$' in self) {\n            // @ts-ignore __webpack_module__ is global\n            var currentExports = module.exports;\n            // @ts-ignore __webpack_module__ is global\n            var prevSignature = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevSignature) !== null && _b !== void 0 ? _b : null;\n            // This cannot happen in MainTemplate because the exports mismatch between\n            // templating and execution.\n            self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.id);\n            // A module can be accepted automatically based on its exports, e.g. when\n            // it is a Refresh Boundary.\n            if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {\n                // Save the previous exports signature on update so we can compare the boundary\n                // signatures. We avoid saving exports themselves since it causes memory leaks (https://github.com/vercel/next.js/pull/53797)\n                module.hot.dispose(function (data) {\n                    data.prevSignature =\n                        self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports);\n                });\n                // Unconditionally accept an update to this module, we'll check if it's\n                // still a Refresh Boundary later.\n                // @ts-ignore importMeta is replaced in the loader\n                module.hot.accept();\n                // This field is set when the previous version of this module was a\n                // Refresh Boundary, letting us know we need to check for invalidation or\n                // enqueue an update.\n                if (prevSignature !== null) {\n                    // A boundary can become ineligible if its exports are incompatible\n                    // with the previous exports.\n                    //\n                    // For example, if you add/remove/change exports, we'll want to\n                    // re-execute the importing modules, and force those components to\n                    // re-render. Similarly, if you convert a class component to a\n                    // function, we want to invalidate the boundary.\n                    if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevSignature, self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports))) {\n                        module.hot.invalidate();\n                    }\n                    else {\n                        self.$RefreshHelpers$.scheduleUpdate();\n                    }\n                }\n            }\n            else {\n                // Since we just executed the code for the module, it's possible that the\n                // new exports made it ineligible for being a boundary.\n                // We only care about the case when we were _previously_ a boundary,\n                // because we already accepted this update (accidental side effect).\n                var isNoLongerABoundary = prevSignature !== null;\n                if (isNoLongerABoundary) {\n                    module.hot.invalidate();\n                }\n            }\n        }\n    })();\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGFwcC1wYWdlcy1icm93c2VyKS8uL3NyYy9hcGlzL2luZGV4LnRzIiwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7QUFBeUI7QUFDVztBQUNJO0FBRWpDLE1BQU1HLGVBQWVILDZDQUFLQSxDQUFDSSxNQUFNLENBQUM7SUFDckNDLFNBQVNDLE9BQU9BLENBQUNDLEdBQUcsQ0FBQ0MsVUFBVSxHQUFHO0lBQ2xDQyxTQUFTO0FBQ2IsR0FBRTtBQUVLLE1BQU1DLGlCQUFpQlYsNkNBQUtBLENBQUNJLE1BQU0sQ0FBQztJQUN2Q0MsU0FBU0MsT0FBT0EsQ0FBQ0MsR0FBRyxDQUFDQyxVQUFVLEdBQUc7SUFDbENDLFNBQVM7QUFDYixHQUFFO0FBRWEsU0FBU0UsT0FBT0MsSUFBWTtJQUN2Q0EsT0FBT0EsS0FBS0MsS0FBSyxDQUFDO0lBQ2xCLE1BQU1DLE1BQU1GLElBQUksQ0FBQyxFQUFFO0lBQ25CLE1BQU1HLFFBQVFILElBQUksQ0FBQyxFQUFFO0lBQ3JCSSxRQUFRQyxHQUFHLENBQUNDLFNBQVNIO0lBRXJCLElBQUlELE9BQU8sU0FBUztRQUNoQixJQUFJQyxTQUFTLE9BQ1QsT0FBT2QsOENBQVVBLENBQUNrQixRQUFRO2FBQ3pCLElBQUlKLFNBQVMsUUFDZCxPQUFPZCw4Q0FBVUEsQ0FBQ21CLFNBQVM7YUFDMUIsSUFBSUwsU0FBUyxVQUNkLE9BQU9kLDhDQUFVQSxDQUFDb0IsV0FBVzthQUM1QixJQUFJTixTQUFTLHVCQUNkLE9BQU9kLDhDQUFVQSxDQUFDcUIsd0JBQXdCO2FBQ3pDLElBQUtQLFNBQVMscUJBQ2YsT0FBT2QsOENBQVVBLENBQUNzQixzQkFBc0I7SUFDaEQsT0FDSyxJQUFJVCxPQUFPLFdBQVc7UUFDdkIsSUFBSUMsU0FBUyxPQUNULE9BQU9iLGtEQUFZQSxDQUFDaUIsUUFBUTthQUMzQixJQUFJSixTQUFTLFFBQ2QsT0FBT2Isa0RBQVlBLENBQUNrQixTQUFTO2FBQzVCLElBQUlMLFNBQVMsVUFDZCxPQUFPYixrREFBWUEsQ0FBQ21CLFdBQVc7YUFDOUIsSUFBSU4sU0FBUyx1QkFDZCxPQUFPYixrREFBWUEsQ0FBQ29CLHdCQUF3QjthQUMzQyxJQUFLUCxTQUFTLHFCQUNmLE9BQU9iLGtEQUFZQSxDQUFDcUIsc0JBQXNCO0lBQ2xEO0lBRUEsT0FBTztBQUNYIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vX05fRS8uL3NyYy9hcGlzL2luZGV4LnRzPzRmZGYiXSwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IGF4aW9zIGZyb20gJ2F4aW9zJ1xuaW1wb3J0IHsgdXNpbmdHTE9WRSB9IGZyb20gJy4vZ2xvdmUnXG5pbXBvcnQgeyB1c2luZ1BIT0JFUlQgfSBmcm9tICcuL3Bob2JlcnQnXG5cbmV4cG9ydCBjb25zdCBHTE9WRV9DT05GSUcgPSBheGlvcy5jcmVhdGUoe1xuICAgIGJhc2VVUkw6IHByb2Nlc3MuZW52LlNFUlZFUl9BUEkgKyAnL2dsb3ZlJyxcbiAgICB0aW1lb3V0OiAxMDAwMFxufSlcblxuZXhwb3J0IGNvbnN0IFBIT0JFUlRfQ09ORklHID0gYXhpb3MuY3JlYXRlKHtcbiAgICBiYXNlVVJMOiBwcm9jZXNzLmVudi5TRVJWRVJfQVBJICsgJy9waG9iZXJ0JyxcbiAgICB0aW1lb3V0OiAxMDAwMFxufSlcblxuZXhwb3J0IGRlZmF1bHQgZnVuY3Rpb24gdXNlQVBJKHBhdGg6IHN0cmluZykge1xuICAgIHBhdGggPSBwYXRoLnNwbGl0KCcnKVxuICAgIGNvbnN0IGVtYiA9IHBhdGhbMl1cbiAgICBjb25zdCBtb2RlbCA9IHBhdGhbM11cbiAgICBjb25zb2xlLmxvZyhwYXRoZW1iLCBtb2RlbClcblxuICAgIGlmIChlbWIgPT0gJ2dsb3ZlJykge1xuICAgICAgICBpZiAobW9kZWwgPT0gJ2NubicpIFxuICAgICAgICAgICAgcmV0dXJuIHVzaW5nR0xPVkUudXNpbmdDTk5cbiAgICAgICAgZWxzZSBpZiAobW9kZWwgPT0gJ2xzdG0nKVxuICAgICAgICAgICAgcmV0dXJuIHVzaW5nR0xPVkUudXNpbmdMU1RNXG4gICAgICAgIGVsc2UgaWYgKG1vZGVsID09ICdiaWxzdG0nKVxuICAgICAgICAgICAgcmV0dXJuIHVzaW5nR0xPVkUudXNpbmdCSUxTVE1cbiAgICAgICAgZWxzZSBpZiAobW9kZWwgPT0gJ2Vuc2VtYmxlX2Nubl9iaWxzdG0nKVxuICAgICAgICAgICAgcmV0dXJuIHVzaW5nR0xPVkUudXNpbmdFTlNFTUJMRV9DTk5fQklMU1RNXG4gICAgICAgIGVsc2UgaWYgKCBtb2RlbCA9PSAnZnVzaW9uX2Nubl9iaWxzdG0nKVxuICAgICAgICAgICAgcmV0dXJuIHVzaW5nR0xPVkUudXNpbmdGVVNJT05fQ05OX0JJTFNUTVxuICAgIH1cbiAgICBlbHNlIGlmIChlbWIgPT0gJ3Bob2JlcnQnKSB7XG4gICAgICAgIGlmIChtb2RlbCA9PSAnY25uJykgXG4gICAgICAgICAgICByZXR1cm4gdXNpbmdQSE9CRVJULnVzaW5nQ05OXG4gICAgICAgIGVsc2UgaWYgKG1vZGVsID09ICdsc3RtJylcbiAgICAgICAgICAgIHJldHVybiB1c2luZ1BIT0JFUlQudXNpbmdMU1RNXG4gICAgICAgIGVsc2UgaWYgKG1vZGVsID09ICdiaWxzdG0nKVxuICAgICAgICAgICAgcmV0dXJuIHVzaW5nUEhPQkVSVC51c2luZ0JJTFNUTVxuICAgICAgICBlbHNlIGlmIChtb2RlbCA9PSAnZW5zZW1ibGVfY25uX2JpbHN0bScpXG4gICAgICAgICAgICByZXR1cm4gdXNpbmdQSE9CRVJULnVzaW5nRU5TRU1CTEVfQ05OX0JJTFNUTVxuICAgICAgICBlbHNlIGlmICggbW9kZWwgPT0gJ2Z1c2lvbl9jbm5fYmlsc3RtJylcbiAgICAgICAgICAgIHJldHVybiB1c2luZ1BIT0JFUlQudXNpbmdGVVNJT05fQ05OX0JJTFNUTVxuICAgIH1cblxuICAgIHJldHVybiBudWxsXG59Il0sIm5hbWVzIjpbImF4aW9zIiwidXNpbmdHTE9WRSIsInVzaW5nUEhPQkVSVCIsIkdMT1ZFX0NPTkZJRyIsImNyZWF0ZSIsImJhc2VVUkwiLCJwcm9jZXNzIiwiZW52IiwiU0VSVkVSX0FQSSIsInRpbWVvdXQiLCJQSE9CRVJUX0NPTkZJRyIsInVzZUFQSSIsInBhdGgiLCJzcGxpdCIsImVtYiIsIm1vZGVsIiwiY29uc29sZSIsImxvZyIsInBhdGhlbWIiLCJ1c2luZ0NOTiIsInVzaW5nTFNUTSIsInVzaW5nQklMU1RNIiwidXNpbmdFTlNFTUJMRV9DTk5fQklMU1RNIiwidXNpbmdGVVNJT05fQ05OX0JJTFNUTSJdLCJzb3VyY2VSb290IjoiIn0=\n//# sourceURL=webpack-internal:///(app-pages-browser)/./src/apis/index.ts\n"));

/***/ })

});