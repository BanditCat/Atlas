;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                    Copyright (c) Jon DuBois 2024.                          ;;
;;                                                                            ;;
;; This program is free software: you can redistribute it and/or modify       ;;
;; it under the terms of the GNU Affero General Public License as published   ;;
;; by the Free Software Foundation, either version 3 of the License, or       ;;
;; (at your option) any later version.                                        ;;
;;                                                                            ;;
;; This program is distributed in the hope that it will be useful,            ;;
;; but WITHOUT ANY WARRANTY; without even the implied warranty of             ;;
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              ;;
;; GNU Affero General Public License for more details.                        ;;
;;                                                                            ;;
;; You should have received a copy of the GNU Affero General Public License   ;;
;; along with this program.  If not, see <https://www.gnu.org/licenses/>.     ;;
;;                                                                            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(set-language-environment "UTF-8")
(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(ansi-color-faces-vector
   [default default default italic underline success warning error])
 '(ansi-color-names-vector
   ["#212526" "#ff4b4b" "#b4fa70" "#fce94f" "#729fcf" "#e090d7" "#8cc4ff" "#eeeeec"])
 '(cua-mode t nil (cua-base))
 '(custom-enabled-themes '(wheatgrass))
 '(desktop-save-mode t)
 '(global-tab-line-mode t)
 '(package-archives
   '(("gnu" . "https://elpa.gnu.org/packages/")
     ("melpa" . "https://melpa.org/packages/")))
 '(package-selected-packages '(haskell-mode))
 '(shell-file-name "C:\\Program Files\\Git\\bin\\bash.exe")
 '(show-paren-mode t)
 '(tab-line-close-button-show t)
 '(tab-line-new-button-show nil)
 '(tab-line-tabs-function 'tab-line-tabs-window-buffers))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(default ((t (:family "APL385 Unicode" :foundry "outline" :slant normal :weight normal :height 110 :width normal))))
 '(tab-line ((t (:inherit variable-pitch :background "black" :foreground "light gray" :height 0.9))))
 '(tab-line-highlight ((t (:inherit tab-line-tab :background "midnight blue"))))
 '(tab-line-tab ((t (:inherit tab-line :foreground "white" :box (:line-width 1 :color "white" :style released-button) :height 1.5 :width normal))))
 '(tab-line-tab-current ((t (:inherit tab-line-tab :background "medium blue"))))
 '(tab-line-tab-inactive ((t (:inherit tab-line-tab :background "black")))))

(set-face-attribute 'default nil :height 87)

(defun bandit-run (&optional arg) (interactive)
       (bandit-save)
       (cond
	((eq major-mode 'gnu-apl-mode) (gnu-apl-interactive-send-buffer))
	((eq major-mode 'emacs-lisp-mode) (eval-buffer) (message "ran"))
	((eq major-mode 'python-mode) (async-shell-command (concat "python " (buffer-file-name))))
	((eq major-mode 'haskell-mode) (async-shell-command (concat "runghc " (buffer-file-name))))
	(t (compile (concat "make -j 24 " arg)))))

(defun bandit-copypaste () (interactive)
       (if mark-active (kill-region (mark) (point))
	 (yank)))
(defun bandit-save () (interactive)
       (save-some-buffers t)
       (message "Saved all buffers"))
(defun bandit-kill () (interactive)
       (kill-buffer (current-buffer)))
(defun bandit-indent () (interactive)
       (save-excursion
	 (indent-region (point-min) (point-max) nil)))

(defun bandit-next-window () (interactive)
       (select-window (next-window)))
(defun bandit-previous-window () (interactive)
       (select-window (previous-window)))
(defun bandit-next-tab () (interactive)
       (if (/= 1 (length (tab-line-tabs-window-buffers)))
	   (tab-line-select-tab-buffer
	    (elt (tab-line-tabs-window-buffers)
		 (mod (+ 1 (seq-position (tab-line-tabs-window-buffers) (current-buffer))) (length (tab-line-tabs-window-buffers)))))))
(defun bandit-previous-tab () (interactive)
       (if (/= 1 (length (tab-line-tabs-window-buffers)))
	   (tab-line-select-tab-buffer
	    (elt (tab-line-tabs-window-buffers)
		 (mod (+ (- (length (tab-line-tabs-window-buffers)) 1)
			 (seq-position (tab-line-tabs-window-buffers) (current-buffer))) (length (tab-line-tabs-window-buffers)))))))
(defun bandit-kill-nonfile-buffers () (interactive)
       (mapc
	(lambda (buf)
	  (if (and (not (string-match "\*tab" (buffer-name buf) ))
		   (eq (buffer-file-name buf) nil))
	      (kill-buffer buf)))		     
	(buffer-list)))

(add-hook 'kill-emacs-hook 'bandit-kill-nonfile-buffers -100)

(global-set-key (kbd "<f11>") 'bandit-kill-nonfile-buffers)


(global-set-key (kbd "<kp-decimal>") 'bandit-copypaste)
(global-set-key (kbd "C-s") 'bandit-save)  
(global-set-key (kbd "<mouse-2>") 'shell)
(global-set-key (kbd "<mouse-4>") 'bandit-previous-tab)
(global-set-key (kbd "<mouse-5>") 'bandit-next-tab)
(global-set-key (kbd "<M-right>") 'bandit-next-tab)
(global-set-key (kbd "<M-left>") 'bandit-previous-tab)
(global-set-key (kbd "<M-up>") 'bandit-previous-window)
(global-set-key (kbd "<M-down>") 'bandit-next-window)
(global-set-key (kbd "<f12>") 'bandit-kill)
(global-set-key (kbd "<f5>") 'bandit-run)
(global-set-key (kbd "<f1>") 'find-file)
(global-set-key (kbd "C-a") 'mark-whole-buffer)
(global-set-key (kbd "<f2>") 'ispell-comments-and-strings)
(global-set-key (kbd "<S-f2>") 'ispell-buffer)
(global-set-key (kbd "<f3>") 'bandit-indent)
(global-set-key (kbd "<f4>") 'next-error)
(global-set-key (kbd "<f6>") (lambda () (interactive) (bandit-run "rall")))
(global-set-key (kbd "<S-f6>") (lambda () (interactive) (bandit-run "rdall")))
(global-set-key (kbd "<f7>") (lambda () (interactive) (bandit-run "clean")))
(global-set-key (kbd "<f8>") (lambda () (interactive) (bandit-run "backup")))
(global-set-key (kbd "<f9>") (lambda () (interactive)
			       (insert (mark) (point))))
(global-set-key (kbd "<f10>") (lambda () (interactive) (uncomment-region (markk) (point))))
(global-set-key (kbd "C-l") (kbd "<super>"))

(global-set-key (kbd "C-f") 'isearch-forward-regexp)
(define-key isearch-mode-map "\C-f" 'isearch-repeat-forward)
(global-set-key (kbd "C-S-f") (lambda () (interactive) (multi-occur-in-matching-buffers ".*" (read-string "Regexp:"))))
(global-set-key (kbd "C-S-r") 'query-replace-regexp)


(add-to-list 'exec-path "C:/root/bin/")

(setq ispell-program-name (locate-file "hunspell"
				       exec-path exec-suffixes 'file-executable-p))

(require 'ispell)
(defun next-read-file-uses-dialog-p () t)

(setq display-buffer-base-action
      (cons 'display-buffer-use-some-window
            '((inhibit-same-window . t)
              (inhibit-switch-frame . nil))))

(setq backup-directory-alist
      `((".*" . "~/backups" )))

;(load-library "glsl-mode")
(require 'package)


;(require 'hs-lint)
;(defun my-haskell-mode-hook ()
;   (local-set-key "\C-cl" 'hs-lint))
;(add-hook 'haskell-mode-hook 'my-haskell-mode-hook)

(add-to-list 'load-path "C:\\root\\home\\gnu-apl-mode-master\\")
(require 'gnu-apl-mode)

(package-initialize)

